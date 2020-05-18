precision mediump float;

// Expected to be constant across the entire scene.
layout(row_major, std140) uniform ub_SceneParams {
    Mat4x4 u_Projection;
    float u_Time;
};

layout(row_major, std140) uniform ub_DrawParams {
    Mat4x4 u_Model;
    Mat4x3 u_View;
};

uniform sampler2D u_Texture;

varying vec4 v_Color;
varying vec2 v_TexCoord;

#ifdef VERT
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec4 a_Color;
layout(location = 2) in vec2 a_TexCoord;

void main() {
    gl_Position = Mul(u_Projection, Mul(_Mat4x4(u_View), Mul(u_Model, vec4(a_Position, 1.0))));
    v_Color = a_Color;
    v_TexCoord = a_TexCoord;
}
#endif

#ifdef FRAG
void main() {
    vec4 t_Color = vec4(0.5, 0.5, 0.5, 1);

#ifdef USE_TEXTURE
    t_Color = texture(u_Texture, v_TexCoord);
#endif

#ifdef USE_VERTEX_COLOR
    t_Color.rgb *= v_Color.rgb * 2.0f;
    t_Color.a *= v_Color.a;
#endif

#ifdef USE_ALPHA_MASK
    if (t_Color.a < 0.01) {
        discard;
    }
#else
    if (t_Color.a < 0.15) {
        discard;
    }
#endif

    gl_FragColor = t_Color;
}
#endif
