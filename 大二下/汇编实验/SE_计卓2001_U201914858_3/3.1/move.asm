.386     
.model flat, stdcall
extern HIGHF:sdword, MIDF:sdword, LOWF:sdword

STATUS  STRUCT
    SAMID  DB 12 DUP(0)  
    SDA   DD  256809    
    SDB   DD  -1023    
    SDC   DD   1265    
    SF    DD   0     
STATUS  ENDS

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
        ret
move_low endp

END