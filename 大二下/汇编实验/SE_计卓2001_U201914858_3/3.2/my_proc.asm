.386     
.model flat, stdcall

.CODE
judge proc stdcall A:dword, B:dword, cc:dword
    mov eax, A
    imul eax, 5
    add eax, B
    sub eax, cc
    add eax, 100
    mov edx, 0
    mov ecx, 128
    idiv ecx
    

    ret
judge endp
END
