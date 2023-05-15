 .686     
.model flat, stdcall
 ExitProcess PROTO STDCALL :DWORD
 includelib  kernel32.lib  ; ExitProcess 在 kernel32.lib中实现
 printf          PROTO C :VARARG
 scanf           PROTO C:VARARG
 includelib  libcmt.lib
 includelib  legacy_stdio_definitions.lib

.DATA

SAMID DB 9 DUP(0)   ;每组数据的流水号（可以从1开始编号）
SDA   DD  256809     ;状态信息a
SDB   DD  -1023      ;状态信息b
SDC   DD   1265      ;状态信息c
SF    DD   ?      ;处理结果f

Fmt1  DB "%d",0ah,0dh,0

LOWF  DD 4 DUP(0)
MIDF  DD 4 DUP(0)
HIGHF DD 4 DUP(0)

.STACK 200

.CODE
main proc c
    mov eax, SDA
    imul eax, 5
    add eax, SDB
    sub eax, SDC
    add eax, 100
    mov edx, 0
    mov ecx, 128
    idiv ecx
    cmp eax, 100
    js TOLOWF
    je TOMIDF
    jns TOHIGHF

TOLOWF:
    mov ebx, SDA
    mov LOWF, ebx
    mov ebx, SDB
    mov LOWF[4], ebx
    mov ebx, SDC
    mov LOWF[8], ebx
    mov LOWF[12], eax 
    invoke printf, offset Fmt1, offset LOWF
    invoke ExitProcess, 0

TOMIDF:
    mov ebx, SDA
    mov MIDF, ebx
    mov ebx, SDB
    mov MIDF[4], ebx
    mov ebx, SDC
    mov MIDF[8], ebx
    mov MIDF[12], eax 
    invoke printf, offset Fmt1, MIDF
    invoke printf, offset Fmt1, MIDF+4
    invoke printf, offset Fmt1, MIDF+8
    invoke printf, offset Fmt1, MIDF+12
    invoke ExitProcess, 0

TOHIGHF:
    mov ebx, SDA
    mov HIGHF, ebx
    mov ebx, SDB
    mov HIGHF[4], ebx
    mov ebx, SDC
    mov HIGHF[8], ebx
    mov HIGHF[12], eax 
    invoke printf, offset Fmt1, HIGHF
    invoke printf, offset Fmt1, HIGHF+4
    invoke printf, offset Fmt1, HIGHF+8
    invoke printf, offset Fmt1, HIGHF+12

    invoke ExitProcess, 0

main endp
END