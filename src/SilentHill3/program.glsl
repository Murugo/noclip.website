precision mediump float;

// Expected to be constant across the entire scene.
layout(row_major, std140) uniform ub_SceneParams {
    Mat4x4 u_Projection;
    vec4 u_LightPos;
    vec4 u_LightDir;
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
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec4 a_Color;
layout(location = 3) in vec2 a_TexCoord;

void main() {
    vec4 t_WorldPosition = Mul(u_Model, vec4(a_Position, 1.0));
    vec3 t_WorldNormal = normalize(Mul(u_Model, vec4(a_Normal, 0.0)).xyz);
    gl_Position = Mul(u_Projection, Mul(_Mat4x4(u_View), t_WorldPosition));

    // Temporary "flashlight" for better viewing experience.
    float t_LightIntensity = clamp(2.0 - length(t_WorldPosition - u_LightPos) / 200.0, 0.0, 2.0);
    t_LightIntensity *= clamp(dot(u_LightDir.xyz, normalize((u_LightPos - t_WorldPosition).xyz)) * 3.0 - 2.0, 0.0, 1.0);
    
    v_Color.rgb = max(a_Color.rgb, vec3(t_LightIntensity) * dot(-t_WorldNormal, u_LightDir.xyz));
    v_Color.a = a_Color.a;
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
    t_Color *= v_Color;
#endif

#ifdef USE_ALPHA_MASK
    if (t_Color.a < 0.5) {
        discard;
    }
#endif

    gl_FragColor = t_Color;
}
#endif
