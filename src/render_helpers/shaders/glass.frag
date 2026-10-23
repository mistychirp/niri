#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

uniform float niri_scale;

uniform vec2 geo_size;
uniform vec4 corner_radius;
uniform mat3 input_to_geo;
// Inverse of input_to_geo, to map refracted geometry coords back to texture coords.
uniform mat3 geo_to_input;

// Width of the curved rim, in logical pixels.
uniform float bezel;
// How far the rim bends the background inward at its outer edge, in logical pixels.
uniform float refraction;
// Strength of the specular highlight along the rim.
uniform float highlight;
// Premultiplied tint layered over the glass.
uniform vec4 glass_tint;

vec4 postprocess(vec4 color);

// Signed distance to a rounded box centered at the origin, with per-corner radii
// (x: top-left, y: top-right, z: bottom-right, w: bottom-left; y points down).
float sd_rounded_box(vec2 p, vec2 half_size, vec4 radii) {
    float r = p.x < 0.0
        ? (p.y < 0.0 ? radii.x : radii.w)
        : (p.y < 0.0 ? radii.y : radii.z);
    vec2 q = abs(p) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

vec4 sample_geo(vec2 geo) {
    return texture2D(tex, (geo_to_input * vec3(geo, 1.0)).xy);
}

void main() {
    vec2 geo = (input_to_geo * vec3(v_coords, 1.0)).xy;
    vec2 half_size = geo_size * 0.5;
    vec2 p = geo * geo_size - half_size;

    float d = sd_rounded_box(p, half_size, corner_radius);

    // Antialiased coverage over one physical pixel.
    float coverage = clamp(0.5 - d * niri_scale, 0.0, 1.0);
    if (coverage <= 0.0) {
        gl_FragColor = vec4(0.0);
        return;
    }

    // Outward surface normal from the SDF gradient.
    vec2 e = vec2(0.5, 0.0);
    vec2 n = vec2(
        sd_rounded_box(p + e.xy, half_size, corner_radius) - sd_rounded_box(p - e.xy, half_size, corner_radius),
        sd_rounded_box(p + e.yx, half_size, corner_radius) - sd_rounded_box(p - e.yx, half_size, corner_radius)
    );
    n = length(n) > 0.0 ? normalize(n) : vec2(0.0);

    // 0 at the outer edge, 1 where the rim meets the flat top.
    float t = clamp(-d / max(bezel, 0.001), 0.0, 1.0);
    // A convex rim is steepest at the edge, so it bends light the most there.
    float bend = refraction * pow(1.0 - t, 2.5);

    // Light refracts toward the center, so the rim shows content from further in.
    // Channels bend slightly differently for a hint of dispersion.
    vec2 offset = -n * bend / geo_size;
    vec4 color;
    color.r = sample_geo(geo + offset * 1.06).r;
    color.ga = sample_geo(geo + offset).ga;
    color.b = sample_geo(geo + offset * 0.94).b;

    color = postprocess(color);

    // Tint, premultiplied over the background.
    color = glass_tint + color * (1.0 - glass_tint.a);

    // Specular highlight: light from the top-left catches the rim, plus a faint
    // bounce on the opposite side.
    vec2 light = normalize(vec2(-1.0, -1.4));
    float facing = dot(n, light);
    float rim = pow(1.0 - t, 3.0);
    float spec = rim * (max(facing, 0.0) + 0.35 * max(-facing, 0.0));
    // A crisp one-pixel line right at the edge.
    float edge = clamp(1.0 - (-d * niri_scale), 0.0, 1.0);
    color.rgb += vec3(highlight * (spec + 0.5 * edge)) * color.a;

    color = color * coverage * alpha;

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
