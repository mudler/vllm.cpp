// Scheduled Cortex-A76 Q8_0 x Q8_0 VecDotFn. AAPCS64 leaf, no stack frame on
// the valid hot path. Invalid arguments tail-branch to the C++ throw helper.
// Two blocks are kept in flight so load/SDOT/FP16-convert latency can overlap;
// the two scalar contributions are still added in original block order.
    .arch armv8.2-a+dotprod+fp16
    .text
    .p2align 4
    .global vt_cpu_q8_dot_a76_asm
    .type vt_cpu_q8_dot_a76_asm, %function
vt_cpu_q8_dot_a76_asm:
    .cfi_startproc
    tst     w0, #31
    b.ne    .Lbad_args
    cmp     w7, #1
    b.ne    .Lbad_args
    mov     x9, x1
    mov     x10, x3
    mov     x11, x5
    asr     w2, w0, #5
    fmov    s31, wzr
    cmp     w2, #0
    b.le    .Ldone
    cmp     w2, #2
    b.lt    .Ltail

.Lpairs:
    ldur    q0, [x10, #2]
    ldur    q1, [x11, #2]
    movi    v4.4s, #0
    ldur    q16, [x10, #36]
    sdot    v4.4s, v0.16b, v1.16b
    ldur    q17, [x11, #36]
    movi    v20.4s, #0
    ldur    q2, [x10, #18]
    sdot    v20.4s, v16.16b, v17.16b
    ldur    q3, [x11, #18]
    sdot    v4.4s, v2.16b, v3.16b
    ldur    q18, [x10, #52]
    ldur    q19, [x11, #52]
    ldr     h21, [x10, #34]
    sdot    v20.4s, v18.16b, v19.16b
    ldr     h22, [x11, #34]
    ldr     h5, [x10], #68
    ldr     h6, [x11], #68

    fcvt    s5, h5
    fcvt    s6, h6
    addv    s4, v4.4s
    scvtf   s4, s4
    fmul    s5, s5, s6
    fmul    s4, s4, s5

    addv    s20, v20.4s
    fcvt    s21, h21
    fcvt    s22, h22
    scvtf   s20, s20
    fmul    s21, s21, s22
    fmul    s20, s20, s21
    fadd    s31, s31, s4
    fadd    s31, s31, s20

    sub     w2, w2, #2
    cmp     w2, #2
    b.ge    .Lpairs

.Ltail:
    cbz     w2, .Ldone
    ldur    q0, [x10, #2]
    ldur    q1, [x11, #2]
    movi    v4.4s, #0
    ldur    q2, [x10, #18]
    sdot    v4.4s, v0.16b, v1.16b
    ldur    q3, [x11, #18]
    ldr     h5, [x10]
    sdot    v4.4s, v2.16b, v3.16b
    ldr     h6, [x11]
    addv    s4, v4.4s
    fcvt    s5, h5
    fcvt    s6, h6
    scvtf   s4, s4
    fmul    s5, s5, s6
    fmul    s4, s4, s5
    fadd    s31, s31, s4

.Ldone:
    str     s31, [x9]
    ret

.Lbad_args:
    mov     w1, w7
    b       vt_cpu_q8_dot_a76_bad_args
    .cfi_endproc
    .size vt_cpu_q8_dot_a76_asm, .-vt_cpu_q8_dot_a76_asm
    .section .note.GNU-stack,"",%progbits
