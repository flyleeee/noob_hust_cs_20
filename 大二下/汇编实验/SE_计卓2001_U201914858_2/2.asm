.686     
.model flat, stdcall
ExitProcess PROTO STDCALL :DWORD
includelib  kernel32.lib  ; ExitProcess 在 kernel32.lib中实现
printf          PROTO C :VARARG
scanf           PROTO C:VARARG
clock			PROTO C :VARARG	
includelib  libcmt.lib
includelib  legacy_stdio_definitions.lib

STATUS  STRUCT
    SAMID  DB 12 DUP(0)  
    SDA   DD  256809    
    SDB   DD  -1023    
    SDC   DD   1265    
    SF    DD   0     
STATUS  ENDS

.DATA

S1 STATUS <"000000001", 418479, 423908, 341, 0>
S2 STATUS <"000000002", 847931, -423908, 11, 0>
S3 STATUS <"000000003", 418931, -423908, 21, 0>
S4 STATUS <"000000004", 418471, 423908, 211, 0>
S5 STATUS <"000000005", 0, 12700, 0, 0>
S6 STATUS <"000000006", 0, 12700, 0, 0>
S7 STATUS <"000000007", 1, 1, 1, 1>
N DW 6

Fmt1  DB "%d",0ah,0dh,0
Fmt2  DB "%c",0ah,0dh,0
Fmt3  DB "%c",0
Fmt4  DB "Time the program spent: %d", 0ah, 0dh, 0

LOWF  DD 280 DUP(0)
MIDF  DD 280 DUP(0)
HIGHF DD 280 DUP(0)

begin_time 	DD 0
end_time    DD 0
spend_time 	DD 0

.STACK 200

.CODE
main proc c
MOV EBX, OFFSET S1
MOV ESI, -1
sub ebx, type STATUS

LOOP1:
    inc esi
    add ebx, type STATUS
    JMP MOVE
LOOP2:
    CMP SI, N
    JS LOOP1
    JE EXITP

MOVE:
    invoke clock				;调用clock函数,开始计时
    mov begin_time,eax
    mov ecx, 10000000           ;每条信息判断一千万次

    LOOPMAX:
    push ecx
    mov eax, (STATUS PTR [EBX]).SDA
    LEA EAX,[EAX+EAX*4+100]		    ;实现a*5+100
    ADD EAX,(STATUS PTR [EBX]).SDB	
    SUB EAX,(STATUS PTR [EBX]).SDC	
    ;imul eax, 5
    ;add eax, (STATUS PTR [EBX]).SDB
    ;sub eax, (STATUS PTR [EBX]).SDC
    ;add eax, 100
    sar eax, 7
    ;mov edx, 0
    ;mov ecx, 128
    ;idiv ecx
    mov (STATUS PTR [EBX]).SF, eax
    cmp eax, 100
    pop ecx

    loop LOOPMAX
    invoke clock
    sub eax, begin_time
    mov spend_time, eax
	invoke printf, offset Fmt4, eax	;打印所用时间
    js TOLOWF
    je TOMIDF
    jns TOHIGHF

EXITP:
    invoke ExitProcess, 0

TOLOWF:
    imul edi, esi, type STATUS
    mov eax, 0
LOOP_LOW1:
    mov dl, byte ptr [ebx][eax]
    mov byte ptr LOWF[edi][eax], dl
    inc eax
    cmp eax, 12
    jne LOOP_LOW1
    mov edx, (STATUS PTR [EBX]).SDA
    mov LOWF[edi][12], edx
    mov edx, (STATUS PTR [EBX]).SDB
    mov LOWF[edi][16], edx
    mov edx, (STATUS PTR [EBX]).SDC
    mov LOWF[edi][20], edx
    mov edx, (STATUS PTR [EBX]).SF
    mov LOWF[edi][24], edx
    mov edx, 0
LOOP_LOW2:
    push edx
    invoke printf, offset Fmt3, byte ptr LOWF[edi][edx]
    pop edx
    inc edx
    cmp edx, 12
    jne LOOP_LOW2
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, LOWF[edi][12]
    invoke printf, offset Fmt1, LOWF[edi][16]
    invoke printf, offset Fmt1, LOWF[edi][20]
    invoke printf, offset Fmt1, LOWF[edi][24]
    invoke printf, offset Fmt2, 0 
    JMP LOOP2

TOMIDF:
    imul edi, esi, type STATUS
    mov eax, 0
LOOP_MID1:
    mov dl, byte ptr [ebx][eax]
    mov byte ptr MIDF[edi][eax], dl
    inc eax
    cmp eax, 12
    jne LOOP_MID1
    mov edx, (STATUS PTR [EBX]).SDA
    mov MIDF[edi][12], edx
    mov edx, (STATUS PTR [EBX]).SDB
    mov MIDF[edi][16], edx
    mov edx, (STATUS PTR [EBX]).SDC
    mov MIDF[edi][20], edx
    mov edx, (STATUS PTR [EBX]).SF
    mov MIDF[edi][24], edx
    mov edx, 0
LOOP_MID2:
    push edx
    invoke printf, offset Fmt3, byte ptr MIDF[edi][edx]
    pop edx
    inc edx
    cmp edx, 12
    jne LOOP_MID2
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, MIDF[edi][12]
    invoke printf, offset Fmt1, MIDF[edi][16]
    invoke printf, offset Fmt1, MIDF[edi][20]
    invoke printf, offset Fmt1, MIDF[edi][24]
    invoke printf, offset Fmt2, 0 
    JMP LOOP2

TOHIGHF:
    imul edi, esi, type STATUS
    mov eax, 0
LOOP3:
    mov dl, byte ptr [ebx][eax]
    mov byte ptr HIGHF[edi][eax], dl
    inc eax
    cmp eax, 12
    jne LOOP3
    mov edx, (STATUS PTR [EBX]).SDA
    mov HIGHF[edi][12], edx
    mov edx, (STATUS PTR [EBX]).SDB
    mov HIGHF[edi][16], edx
    mov edx, (STATUS PTR [EBX]).SDC
    mov HIGHF[edi][20], edx
    mov edx, (STATUS PTR [EBX]).SF
    mov HIGHF[edi][24], edx
    mov edx, 0
LOOP4:
    push edx
    invoke printf, offset Fmt3, byte ptr HIGHF[edi][edx]
    pop edx
    inc edx
    cmp edx, 12
    jne LOOP4
    invoke printf, offset Fmt2, 0 
    invoke printf, offset Fmt1, HIGHF[edi][12]
    invoke printf, offset Fmt1, HIGHF[edi][16]
    invoke printf, offset Fmt1, HIGHF[edi][20]
    invoke printf, offset Fmt1, HIGHF[edi][24]
    invoke printf, offset Fmt2, 0 
    JMP LOOP2

main endp
END