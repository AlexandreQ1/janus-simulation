#version 330 core

// Fullscreen triangle — no vertex buffers required.
// Bind an empty VAO and draw 3 vertices.
//
// Vertex positions (NDC):
//   ID 0: (-1, -1)
//   ID 1: ( 3, -1)
//   ID 2: (-1,  3)
// The triangle covers the entire [-1,1]^2 clip-space square.

out vec2 vTexCoord;

void main() {
    // Pack UV into the lower bits of gl_VertexID
    vec2 tc = vec2(
        float((gl_VertexID << 1) & 2),
        float( gl_VertexID       & 2)
    );

    vTexCoord   = tc;                               // [0,1] dans la zone visible (tc=2 est clippé)
    gl_Position = vec4(tc * 2.0 - 1.0, 0.0, 1.0); // [-1, 1] NDC
}
