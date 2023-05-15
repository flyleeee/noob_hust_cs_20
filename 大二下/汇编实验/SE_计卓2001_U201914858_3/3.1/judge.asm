.386     
.model flat, stdcall

.CODE
judge proc stdcall A:dword, B:dword, cc:dword
    mov eax, A
    lea eax,[eax+eax*4+100]		;实现a*5+100
    add eax, b	
    sub eax, cc
    sar eax, 7                  ;实现a/128
    ;mov edx, 0
    ;mov ecx, 128
    ;idiv ecx

    ret
judge endp
END
