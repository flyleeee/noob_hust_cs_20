.386     
.model flat, stdcall

.CODE
judge proc stdcall A:dword, B:dword, cc:dword
    mov eax, A
    lea eax,[eax+eax*4+100]		; µœ÷a*5+100
    add eax, b	
    sub eax, cc


    ret
judge endp
END
