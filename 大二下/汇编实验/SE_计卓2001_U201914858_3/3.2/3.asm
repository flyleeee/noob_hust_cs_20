.686     
.model flat, stdcall
ExitProcess PROTO STDCALL :DWORD
includelib  kernel32.lib  ; ExitProcess 在 kernel32.lib中实现
printf          PROTO C :VARARG
scanf           PROTO C:VARARG
includelib  libcmt.lib
includelib  legacy_stdio_definitions.lib
judge proto stdcall A:dword, B:dword, cc:dword
STATUS  STRUCT
    SAMID  DB 12 DUP(0)  
    SDA   DD  256809    
    SDB   DD  -1023    
    SDC   DD   1265    
    SF    DD   0     
STATUS  ENDS

.DATA
BNAME DB "wangyibin", 0, 0
BPASS DB "123456789", 0, 0
INAME   DB  12 DUP(0)
IPASS  DB  12 DUP(0)
TIP DB "plz enter your name and password:", 0

S1 STATUS <"000000001", 418479, 423908, -423908, 0>
S2 STATUS <"000000002", 847931, -423908, -423908, 0>
S3 STATUS <"000000003", 418931, -423908, 21, 0>
S4 STATUS <"000000004", 418471, 423908, 211, 0>
N DD 4
MAX_COUNT DD 3

Fmt1  DB "%d", 0ah, 0dh, 0
Fmt2  DB "%c", 0ah, 0dh, 0
Fmt3  DB "%c", 0
Fmt4  DB "%s", 0
Fmt5  DB "%s", 0ah, 0dh, 0

LOWF  DD 300 DUP(0)
MIDF  DD 300 DUP(0)
HIGHF DD 300 DUP(0)

.STACK 200

.CODE

move_high proc stdcall 
        mov eax, 0
    LOOPHIGH1:
        mov dl, byte ptr [ebx][eax]
        mov byte ptr HIGHF[edi][eax], dl
        inc eax
        cmp eax, 12
        jne LOOPHIGH1

        mov edx, (STATUS PTR [EBX]).SDA
        mov HIGHF[edi][12], edx
        mov edx, (STATUS PTR [EBX]).SDB
        mov HIGHF[edi][16], edx
        mov edx, (STATUS PTR [EBX]).SDC
        mov HIGHF[edi][20], edx
        mov edx, (STATUS PTR [EBX]).SF
        mov HIGHF[edi][24], edx

        mov edx, 0
    LOOPHIGH2:
        push edx
        invoke printf, offset Fmt3, byte ptr HIGHF[edi][edx]
        pop edx
        inc edx
        cmp edx, 12
        jne LOOPHIGH2
        ret
move_high endp

move_mid proc stdcall 
        mov eax, 0
    LOOPMID1:
        mov dl, byte ptr [ebx][eax]
        mov byte ptr MIDF[edi][eax], dl
        inc eax
        cmp eax, 12
        jne LOOPMID1

        mov edx, (STATUS PTR [EBX]).SDA
        mov MIDF[edi][12], edx
        mov edx, (STATUS PTR [EBX]).SDB
        mov MIDF[edi][16], edx
        mov edx, (STATUS PTR [EBX]).SDC
        mov MIDF[edi][20], edx
        mov edx, (STATUS PTR [EBX]).SF
        mov MIDF[edi][24], edx

        mov edx, 0
    LOOPMID2:
        push edx
        invoke printf, offset Fmt3, byte ptr MIDF[edi][edx]
        pop edx
        inc edx
        cmp edx, 12
        jne LOOPMID2
        ret
move_mid endp

move_low proc stdcall 
        mov eax, 0
    LOOPLOW1:
        mov dl, byte ptr [ebx][eax]
        mov byte ptr LOWF[edi][eax], dl
        inc eax
        cmp eax, 12
        jne LOOPLOW1

        mov edx, (STATUS PTR [EBX]).SDA
        mov LOWF[edi][12], edx
        mov edx, (STATUS PTR [EBX]).SDB
        mov LOWF[edi][16], edx
        mov edx, (STATUS PTR [EBX]).SDC
        mov LOWF[edi][20], edx
        mov edx, (STATUS PTR [EBX]).SF
        mov LOWF[edi][24], edx

        mov edx, 0
    LOOPLOW2:
        push edx
        invoke printf, offset Fmt3, byte ptr LOWF[edi][edx]
        pop edx
        inc edx
        cmp edx, 12
        jne LOOPLOW2
        ret
move_low endp

print_high proc stdcall
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, HIGHF[edi][12]
    invoke printf, offset Fmt1, HIGHF[edi][16]
    invoke printf, offset Fmt1, HIGHF[edi][20]
    invoke printf, offset Fmt1, HIGHF[edi][24]
    invoke printf, offset Fmt2, 0 
    ret
print_high endp

print_mid proc stdcall
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, MIDF[edi][12]
    invoke printf, offset Fmt1, MIDF[edi][16]
    invoke printf, offset Fmt1, MIDF[edi][20]
    invoke printf, offset Fmt1, MIDF[edi][24]
    invoke printf, offset Fmt2, 0 
    ret
print_mid endp

print_low proc stdcall
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, LOWF[edi][12]
    invoke printf, offset Fmt1, LOWF[edi][16]
    invoke printf, offset Fmt1, LOWF[edi][20]
    invoke printf, offset Fmt1, LOWF[edi][24]
    invoke printf, offset Fmt2, 0 
    ret
print_low endp

strcmp macro s1, s2   ; 注意该宏用完后eax会被改变，所以及时使用完比较结果通过跳栈方式将数字跳出来
    LOCAL NOTEQU, STRCMP_ENDP
    push esi
    push edi
    MOV ESI,s1
    MOV EDI,s2
    MOV ECX,1
    CLD
    REPE CMPSB
    JNZ NOTEQU
    MOV EAX,1
    jmp STRCMP_ENDP
NOTEQU:
    MOV EAX,0
    jmp STRCMP_ENDP
STRCMP_ENDP:
    pop edi
    pop esi
endm

main proc c
    MOV EBX, OFFSET S1
    MOV ESI, -1
    sub ebx, type STATUS
    mov ecx, 0
LOGIN:
    invoke printf, offset Fmt5, OFFSET TIP
    invoke scanf,offset Fmt4,OFFSET INAME
    invoke scanf,offset Fmt4,OFFSET IPASS
    strcmp offset INAME, offset BNAME
    cmp eax, 1
    jne ERROR
    strcmp offset IPASS, offset BPASS
    cmp eax, 1
    jne ERROR
    je LOOP1

ERROR:
    inc ecx
    cmp ecx, MAX_COUNT
    je EXITP
    jne LOGIN
    

LOOP1:
    invoke printf, offset Fmt2, 0 
    inc esi
    add ebx, type STATUS
    invoke judge, (STATUS PTR [EBX]).SDA, (STATUS PTR [EBX]).SDB, (STATUS PTR [EBX]).SDC
    mov (STATUS PTR [EBX]).SF, eax
    cmp eax, 100
    js TOLOWF
    je TOMIDF
    jns TOHIGHF
LOOP2:
    CMP SI, 3
    JS LOOP1
    JE EXITP
   

TOLOWF:
    imul edi, esi, type STATUS
    invoke move_low
    invoke print_low
    JMP LOOP2

TOMIDF:
    imul edi, esi, type STATUS
    invoke move_mid
    invoke print_mid
    JMP LOOP2

TOHIGHF:
    imul edi, esi, type STATUS
    invoke move_high
    invoke print_high
    JMP LOOP2

EXITP:
    invoke ExitProcess, 0


main endp
END