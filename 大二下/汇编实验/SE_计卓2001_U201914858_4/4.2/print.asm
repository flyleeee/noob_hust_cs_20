.386     
.model flat, stdcall
printf          PROTO C :VARARG
includelib  libcmt.lib
includelib  legacy_stdio_definitions.lib
extern HIGHF:sdword, MIDF:sdword, LOWF:sdword, N:sdword, Fmt1:sbyte, Fmt2:sbyte, Fmt3:sbyte, Fmt6:sbyte, Fmt7:sbyte, Fmt8:sbyte, Fmt9:sbyte, Fmt10:sbyte

STATUS  STRUCT
    SAMID  DB 12 DUP(0)  
    SDA   DD  256809    
    SDB   DD  -1023    
    SDC   DD   1265    
    SF    DD   0     
STATUS  ENDS

.CODE

print_mid proc stdcall
    mov edi, 0
    mov esi, -1
    print_mid_start:
    inc esi
    cmp esi, 4
    je print_mid_end
    imul edi, esi, type STATUS
    cmp MIDF[edi][24], 100
    jne print_mid_start
    invoke printf, offset Fmt6, 0 
    mov edx, 0
    
    LOOPMID2:
        push edx
        invoke printf, offset Fmt3, byte ptr MIDF[edi][edx]
        pop edx
        inc edx
        cmp edx, 12
        jne LOOPMID2
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt7, 0 
    invoke printf, offset Fmt1, MIDF[edi][12]
    invoke printf, offset Fmt8, 0 
    invoke printf, offset Fmt1, MIDF[edi][16]
    invoke printf, offset Fmt9, 0 
    invoke printf, offset Fmt1, MIDF[edi][20]
    invoke printf, offset Fmt10, 0 
    invoke printf, offset Fmt1, MIDF[edi][24]
    invoke printf, offset Fmt2, 0 
    jmp print_mid_start
    print_mid_end:
    ret
print_mid endp

END