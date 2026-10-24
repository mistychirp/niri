use std::sync::{Arc, Mutex};

use niri_config::CornerRadius;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::utils::{Logical, Point, Rectangle, Scale};
use smithay::wayland::compositor::{with_states, SurfaceData};
use wayland_server::protocol::wl_surface::WlSurface;

use crate::handlers::background_effect::get_cached_blur_region;
use crate::niri_render_elements;
use crate::protocols::macshell_glass::{self, GlassCachedState};
use crate::render_helpers::blur::BlurOptions;
use crate::render_helpers::damage::ExtraDamage;
use crate::render_helpers::framebuffer_effect::{FramebufferEffect, FramebufferEffectElement};
use crate::render_helpers::xray::{XrayElement, XrayPos};
use crate::render_helpers::RenderCtx;
use crate::utils::region::TransformedRegion;
use crate::utils::surface_geo;

#[derive(Debug)]
pub struct BackgroundEffect {
    nonxray: FramebufferEffect,
    /// Damage when options change.
    damage: ExtraDamage,
    /// Corner radius for clipping.
    ///
    /// Stored here in addition to `RenderParams` to damage when it changes.
    // FIXME: would be good to remove this duplication of radius.
    corner_radius: CornerRadius,
    blur_config: niri_config::Blur,
    options: Options,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Options {
    pub blur: bool,
    pub xray: bool,
    pub noise: Option<f64>,
    pub saturation: Option<f64>,
    pub glass: bool,
}

impl Options {
    fn is_visible(&self) -> bool {
        self.xray
            || self.glass
            || self.blur
            || self.noise.is_some_and(|x| x > 0.)
            || self.saturation.is_some_and(|x| x != 1.)
    }
}

/// Render-time parameters.
#[derive(Debug)]
pub struct RenderParams {
    /// Geometry of the background effect.
    pub geometry: Rectangle<f64, Logical>,
    /// Effect subregion, will be clipped to `geometry`.
    ///
    /// `subregion.iter()` should return `geometry`-relative rectangles.
    pub subregion: Option<TransformedRegion>,
    /// Geometry and radius for clipping in the same coordinate space as `geometry`.
    pub clip: Option<(Rectangle<f64, Logical>, CornerRadius)>,
    /// Scale to use for rounding to physical pixels.
    pub scale: f64,
}

impl RenderParams {
    fn fit_clip_radius(&mut self) {
        if let Some((geo, radius)) = &mut self.clip {
            // HACK: increase radius to avoid slight bleed on rounded corners.
            *radius = radius.expanded_by(1.);

            *radius = radius.fit_to(geo.size.w as f32, geo.size.h as f32);
        }
    }
}

niri_render_elements! {
    BackgroundEffectElement => {
        FramebufferEffect = FramebufferEffectElement,
        Xray = XrayElement,
        ExtraDamage = ExtraDamage,
    }
}

impl BackgroundEffect {
    pub fn new() -> Self {
        Self {
            nonxray: FramebufferEffect::new(),
            damage: ExtraDamage::new(),
            corner_radius: CornerRadius::default(),
            blur_config: niri_config::Blur::default(),
            options: Options::default(),
        }
    }

    /// Damage the background effect, for example when a blur subregion changes.
    pub fn damage(&mut self) {
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn update_config(&mut self, config: niri_config::Blur) {
        if self.blur_config == config {
            return;
        }

        self.blur_config = config;
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn update_render_elements(
        &mut self,
        corner_radius: CornerRadius,
        effect: niri_config::BackgroundEffect,
        has_blur_region: bool,
    ) {
        // If the surface explicitly requests a blur region, default blur to true.
        let blur = if has_blur_region {
            effect.blur != Some(false)
        } else {
            effect.blur == Some(true)
        };

        let mut options = Options {
            blur,
            xray: effect.xray == Some(true),
            noise: effect.noise,
            saturation: effect.saturation,
            glass: effect.glass == Some(true),
        };

        // If we have some background effect but xray wasn't explicitly set, default it to true
        // since it's cheaper.
        if options.is_visible() && effect.xray.is_none() {
            options.xray = true;
        }

        // Glass refracts what's actually behind the surface, which xray can't provide.
        if options.glass {
            options.xray = false;
        }

        if self.options == options && self.corner_radius == corner_radius {
            return;
        }

        self.options = options;
        self.corner_radius = corner_radius;
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn is_visible(&self) -> bool {
        self.options.is_visible()
    }

    pub fn render(
        &self,
        ctx: RenderCtx<GlesRenderer>,
        ns: Option<usize>,
        mut params: RenderParams,
        xray_pos: XrayPos,
        push: &mut dyn FnMut(BackgroundEffectElement),
    ) {
        if !self.is_visible() {
            return;
        }

        if let Some(clip) = &mut params.clip {
            clip.1 = self.corner_radius;
        }
        params.fit_clip_radius();

        let damage = self.damage.render(params.geometry);

        // Use noise/saturation from options, falling back to blur defaults if blurred, and
        // to no effect if not blurred.
        let blur = self.options.blur && !self.blur_config.off;
        let blur_options = blur.then_some(BlurOptions::from(self.blur_config));
        let noise = if blur { self.blur_config.noise } else { 0. };
        let noise = self.options.noise.unwrap_or(noise) as f32;
        let saturation = if blur {
            self.blur_config.saturation
        } else {
            1.
        };
        let saturation = self.options.saturation.unwrap_or(saturation) as f32;

        if self.options.xray {
            let Some(xray) = ctx.xray else {
                return;
            };

            push(damage.into());
            xray.render(
                ctx,
                params,
                xray_pos,
                blur,
                noise,
                saturation,
                &mut |elem| push(elem.into()),
            );
        } else {
            // Render non-xray effect.
            let elem = self
                .nonxray
                .render(
                ns,
                params,
                blur_options,
                noise,
                saturation,
                self.options.glass.then_some(self.blur_config.glass),
                None,
            );
            push(elem.into());
        }
    }
}

fn render_params_for_tile(
    geometry: Rectangle<f64, Logical>,
    scale: f64,
    clip_to_geometry: bool,
    block_out: bool,
    blur_region: Option<Arc<Vec<Rectangle<i32, Logical>>>>,
    surface_geo: Rectangle<f64, Logical>,
    surface_anim_scale: Scale<f64>,
) -> Option<RenderParams> {
    // Effects not requested by the surface itself are drawn to match the geometry.
    let mut clip = true;

    let mut effect_geometry = geometry;
    let mut subregion = None;
    if let Some(rects) = blur_region {
        if rects.is_empty() {
            // Surface has a set, but empty blur region.
            return None;
        } else {
            // If the surface itself requests the effects, apply different defaults.
            clip = clip_to_geometry;

            // Use geometry-shaped blur for blocked-out windows to avoid unintentionally
            // leaking any surface shapes. We render those windows as geometry-shaped solid
            // rectangles anyway.
            if block_out {
                clip = true;
            } else {
                let mut surface_geo = surface_geo.upscale(surface_anim_scale);
                surface_geo.loc += geometry.loc;

                subregion = Some(TransformedRegion {
                    rects,
                    scale: surface_anim_scale,
                    offset: surface_geo.loc,
                });

                surface_geo = surface_geo
                    .to_physical_precise_round(scale)
                    .to_logical(scale);
                effect_geometry = surface_geo;
            }
        }
    }

    // This corner radius is reset to self.corner_radius in render().
    let clip = clip.then_some((geometry, CornerRadius::default()));

    Some(RenderParams {
        geometry: effect_geometry,
        subregion,
        clip,
        scale,
    })
}

/// Per-surface background effect stored in its data map.
struct SurfaceBackgroundEffect(Mutex<BackgroundEffect>);

impl SurfaceBackgroundEffect {
    fn get(states: &SurfaceData) -> &Self {
        states
            .data_map
            .get_or_insert(|| SurfaceBackgroundEffect(Mutex::new(BackgroundEffect::new())))
    }
}

/// Per-surface glass effects from macshell-glass-v1, one per shape.
#[derive(Default)]
struct SurfaceGlass(Mutex<SurfaceGlassInner>);

#[derive(Default)]
struct SurfaceGlassInner {
    effects: Vec<FramebufferEffect>,
    /// Config the effects were last rendered with, to damage them when it changes.
    config: Option<niri_config::Blur>,
}

pub fn damage_glass(states: &SurfaceData) {
    if let Some(glass) = states.data_map.get::<SurfaceGlass>() {
        for effect in &mut glass.0.lock().unwrap().effects {
            effect.damage();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_glass_shapes(
    ns: Option<usize>,
    geometry: Rectangle<f64, Logical>,
    scale: f64,
    states: &SurfaceData,
    surface_geo: Rectangle<f64, Logical>,
    surface_anim_scale: Scale<f64>,
    blur_config: niri_config::Blur,
    push: &mut dyn FnMut(BackgroundEffectElement),
) {
    let mut cached = states.cached_state.get::<GlassCachedState>();
    let shapes = &cached.current().shapes;
    if shapes.is_empty() {
        return;
    }

    let glass = states.data_map.get_or_insert(SurfaceGlass::default);
    let mut glass = glass.0.lock().unwrap();
    let glass = &mut *glass;

    glass.effects.resize_with(shapes.len(), FramebufferEffect::new);
    if glass.config != Some(blur_config) {
        glass.config = Some(blur_config);
        for effect in &mut glass.effects {
            effect.damage();
        }
    }

    // Same transform as blur subregions: surface-local, then animation scale, then position.
    let offset = surface_geo.loc.upscale(surface_anim_scale) + geometry.loc;
    let radius_scale = surface_anim_scale.x.min(surface_anim_scale.y) as f32;

    // Brightness tracking measures behind the first shape.
    let mut probe = macshell_glass::brightness_probe(states);

    for (shape, effect) in shapes.iter().zip(&glass.effects) {
        let rect = shape.rect.to_f64();
        let a = rect.loc.upscale(surface_anim_scale) + offset;
        let b = (rect.loc + rect.size.to_point()).upscale(surface_anim_scale) + offset;
        let geo = Rectangle::from_extremities(a, b)
            .to_physical_precise_round(scale)
            .to_logical(scale);

        let radius = CornerRadius::from(shape.radius * radius_scale)
            .fit_to(geo.size.w as f32, geo.size.h as f32);

        let frosted = shape.frosted && !blur_config.off;
        let frost = blur_config.glass.frost;
        let params = RenderParams {
            geometry: geo,
            subregion: None,
            clip: Some((geo, radius)),
            scale,
        };
        let elem = effect.render(
            ns,
            params,
            frosted.then_some(BlurOptions {
                passes: frost.passes,
                offset: frost.offset,
            }),
            if frosted { frost.noise as f32 } else { 0. },
            if frosted { frost.saturation as f32 } else { 1. },
            Some(blur_config.glass),
            probe.take(),
        );
        push(elem.into());
    }
}

pub fn damage_surface(states: &SurfaceData) {
    if let Some(effect) = states.data_map.get::<SurfaceBackgroundEffect>() {
        effect.0.lock().unwrap().damage();
    }
}

// Silence, Clippy
// A Smithay user is talking
#[allow(clippy::too_many_arguments)]
pub fn render_for_tile(
    ctx: RenderCtx<GlesRenderer>,
    ns: Option<usize>,
    geometry: Rectangle<f64, Logical>,
    scale: f64,
    clip_to_geometry: bool,
    surface: &WlSurface,
    surface_off: Point<f64, Logical>,
    surface_anim_scale: Scale<f64>,
    blur_config: niri_config::Blur,
    radius: CornerRadius,
    effect: niri_config::BackgroundEffect,
    should_block_out: bool,
    xray_pos: XrayPos,
    push: &mut dyn FnMut(BackgroundEffectElement),
) {
    with_states(surface, |states| {
        // Glass requested through macshell-glass-v1 renders regardless of rules.
        if !should_block_out {
            let mut surface_geo = surface_geo(states).unwrap_or_default().to_f64();
            surface_geo.loc += surface_off;
            render_glass_shapes(
                ns,
                geometry,
                scale,
                states,
                surface_geo,
                surface_anim_scale,
                blur_config,
                push,
            );
        }

        let background_effect = SurfaceBackgroundEffect::get(states);
        let mut background_effect = background_effect.0.lock().unwrap();

        let blur_region = get_cached_blur_region(states);
        let has_blur_region = blur_region.as_ref().is_some_and(|r| !r.is_empty());

        background_effect.update_config(blur_config);
        background_effect.update_render_elements(radius, effect, has_blur_region);

        if !background_effect.is_visible() {
            return;
        }

        let mut surface_geo = surface_geo(states).unwrap_or_default().to_f64();
        surface_geo.loc += surface_off;

        let Some(mut params) = render_params_for_tile(
            geometry,
            scale,
            clip_to_geometry,
            should_block_out,
            blur_region,
            surface_geo,
            surface_anim_scale,
        ) else {
            return;
        };

        // Glass needs a real shape for its rim, not the rectangles of a blur region, so it
        // always follows the geometry and its corner radius.
        if effect.glass == Some(true) {
            params.geometry = geometry;
            params.subregion = None;
            params.clip = Some((geometry, CornerRadius::default()));
        }

        let xray_pos = xray_pos.offset(params.geometry.loc - geometry.loc);
        background_effect.render(ctx, ns, params, xray_pos, push);
    });
}
