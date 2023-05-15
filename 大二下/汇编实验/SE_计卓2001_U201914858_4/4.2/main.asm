.686P
.model flat, stdcall
ExitProcess PROTO STDCALL :DWORD
includelib  kernel32.lib  ; ExitProcess 在 kernel32.lib中实现
printf          PROTO C :VARARG
scanf           PROTO C:VARARG
clock			PROTO C :VARARG	
includelib  libcmt.lib
includelib  legacy_stdio_definitions.lib

public HIGHF, MIDF, LOWF, N, Fmt1, Fmt2, Fmt3, Fmt6, Fmt7, Fmt8, Fmt9, Fmt10
VirtualProtect PROTO:DWORD,:DWORD,:DWORD,:DWORD

judge proto stdcall A:dword, B:dword, cc:dword
move_high proto stdcall
move_mid proto stdcall
move_low proto stdcall
print_mid proto stdcall

STATUS  STRUCT
    SAMID  DB 12 DUP(0)  
    SDA   DD  256809    
    SDB   DD  -1023    
    SDC   DD   1265    
    SF    DD   0     
STATUS  ENDS


.DATA
BNAME DB "wangyibin", 0, 0
BPASS DB "1" xor "q","2" xor "q", "3" xor "q", "4" xor "q", "5" xor "q", "6" xor "q", "7" xor "q", "8" xor "q", "9" xor "q", 0, 0 
INAME   DB  9 DUP(0)
IPASS  DB  9 DUP(0)
ICMD DB 3 DUP(0)
RCMD DB "R", 0, 0
QCMD DB "Q", 0, 0
TIP DB "Plz enter your name and password:", 0
TIP2 DB "Login successfully! Welcome!", 0
TIP3 DB "Infomation error!", 0
TIP4 DB "You have no chance!Bye~ ^_^", 0

S1 STATUS <"000000001", -418479, -423908, -423908, 0>
S2 STATUS <"000000002", 2560, 0, 100, 0>
S3 STATUS <"000000003", 418931, -423908, 21, 0>
S4 STATUS <"000000004", 0, 12700, 0, 0>
N DD 4
MAX_COUNT DD 3

Fmt1  DB "%d", 0ah, 0dh, 0
Fmt2  DB "%c", 0ah, 0dh, 0
Fmt3  DB "%c", 0
Fmt4  DB "%s", 0
Fmt5  DB "%s", 0ah, 0dh, 0
Fmt6  DB "SAMID:", 0
Fmt7  DB "SDA:", 0
Fmt8  DB "SDB:", 0
Fmt9  DB "SDC:", 0
Fmt10  DB "SF:", 0


LOWF  DD 300 DUP(1)
MIDF  DD 300 DUP(1)
HIGHF DD 300 DUP(1)

begin_time 	DD 0
end_time    DD 0
spend_time 	DD 0
machine_code db 0c1H, 0f8H, 07H
lenn = $ - machine_code
oldprotect dd ?
.STACK 200

.CODE

strcmp macro s1, s2, len
    LOCAL NOTEQU, STRCMP_ENDP
    push esi
    push edi
    MOV ESI,s1
    MOV EDI,s2
    MOV ECX,len
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
    mov esi, 0
LOGIN: 
    invoke printf, offset Fmt5, OFFSET TIP
    invoke scanf,offset Fmt4,OFFSET INAME
    invoke scanf,offset Fmt4,OFFSET IPASS
    invoke clock
    mov begin_time, eax
    MOV ECX,0
    MOV EDX,OFFSET IPASS
ENCRY:
    xor EAX, EAX
    MOV BL,BYTE PTR [EDX+ECX]
    xor EAX, EAX
    XOR BL,'q'
    xor EAX, EAX
    XCHG BL,BYTE PTR [EDX+ECX]
    xor EAX, EAX
    ADD ECX,1
    CMP ECX,12
    JNE ENCRY
    invoke clock
    sub eax, begin_time
    cmp eax, 100 ; 超时则检查到单步执行，退出
    jg EXITP
    strcmp offset INAME, offset BNAME, 9
    cmp eax, 1
    jne LOGIN_ERROR
    strcmp offset IPASS, offset BPASS, 9
    cmp eax, 1
    jne LOGIN_ERROR
    je LOGIN_SUC

LOGIN_ERROR:
    invoke printf, offset Fmt5, OFFSET TIP3
    inc esi
    cmp esi, 3
    je LOGIN_FAIL
    jne LOGIN

LOGIN_SUC:
    invoke printf, offset Fmt5, OFFSET TIP2

LOOP3:
    MOV EBX, OFFSET S1
    sub ebx, type STATUS
    mov ecx, 0
    mov esi, -1

LOOP1:
    inc esi
    add ebx, type STATUS
    invoke judge, (STATUS PTR [EBX]).SDA, (STATUS PTR [EBX]).SDB, (STATUS PTR [EBX]).SDC
    push ecx
    push esi
    push edi
    push ebx
    push eax
    mov eax,lenn
    mov ebx,40H
    lea ecx,CopyHere
    invoke VirtualProtect,ecx,eax,ebx,offset oldprotect
    mov ecx,lenn
    mov edi,offset CopyHere
    mov esi,offset machine_code
    pop eax
CopyCode:
    mov bl,[esi]
    mov[edi],bl
    inc esi
    inc edi
    loop CopyCode
CopyHere:
    db lenn dup (0);实现a+b-c/128
    pop ebx
    pop edi
    pop esi
    pop ecx
    mov (STATUS PTR [EBX]).SF, eax
    cmp eax, 100
    js TOLOWF
    je TOMIDF
    jns TOHIGHF

LOOP2:
    CMP ESI, N
    JS LOOP1
    JE FINALP

FINALP:
    invoke print_mid
    invoke scanf,offset Fmt4,OFFSET ICMD
    strcmp offset RCMD, offset ICMD, 1
    cmp eax, 1
    je LOOP3
    strcmp offset QCMD, offset ICMD, 1
    cmp eax, 1
    je EXITP
    JMP EXITP

TOLOWF:
    imul edi, esi, type STATUS
    invoke move_low
    JMP LOOP2

TOMIDF:
    imul edi, esi, type STATUS
    invoke move_mid
    JMP LOOP2

TOHIGHF:
    imul edi, esi, type STATUS
    invoke move_high
    JMP LOOP2

LOGIN_FAIL:
    invoke printf, offset Fmt5, OFFSET TIP4
EXITP:

    invoke ExitProcess, 0

main endp
END