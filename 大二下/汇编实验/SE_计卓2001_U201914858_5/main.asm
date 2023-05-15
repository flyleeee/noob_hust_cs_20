.data
  MessageBoxA proto :DWORD ,:DWORD
  extrn ExitProcess : proc  ;编译器不再检查参数是否合规，因此，可以不用PROTO来说明外部函数
  lpContent db  '你好！ 3+6='
  sum       db  0,0
  lpTitle   db  'My first x86-64 Application',0
.code
mainCRTStartup proc     ;这是控制台情况下默认的执行入口点
start proc ;需要在项目“属性-链接-高级”里指定start为入口点,否则会报mainCRTStartup不能解析
   mov  ax,3
   push ax
   mov  ax,6
   push ax
   call  addtwo          ;自己编写的子程序，堆栈传递参数时，可以自行定义协议
   add  al,30h
   mov  sum,al
   ;
   sub  rsp, 28h         
   xor  r9d, r9d
   lea  r8,  lpTitle
   lea  rdx, lpContent
   xor  rcx,rcx
   call MessageBoxA     ;调用系统函数时，则传递参数时需要遵循规定的堆栈操作
   add  rsp,28h
   ;
   sub  rsp, 18h
   mov  rcx,0
   call ExitProcess
start endp
mainCRTStartup endp
;
;求两个16位数的和的子程序
addtwo   proc
    push bx
    mov  ax,[rsp+10]    ; 取堆栈传递过来的参数
    mov  bx,[rsp+12]
    add  ax,bx
    pop  bx
    ret 4     ;
addtwo  endp
;
end
