.386
STACK	SEGMENT	USE16	STACK	;主程序堆栈段
		DB 200 DUP(0)
STACK	ENDS
;
CODE	SEGMENT	USE16
		ASSUME	CS:CODE,DS:CODE,SS:STACK
		HOUR	DB	?,?,':'		;时的ASCII码
		MIN		DB	?,?,':'		;分的ASCII码
		SEC		DB	?,?			;秒的ASCII码
BEGIN:		
		MOV		AL,4		;4是“时”信息的偏移地址
		OUT		70H,AL		;设定将要访问的单元是偏移值为4的“时”信息
		JMP		$+2			;延时，保证端口操作的可靠性
		IN		AL,71H		;读取“时”信息
		MOV		AH,AL		;将2位压缩的BCD码转换为未压缩的BCD码
		AND		AL,0FH		
		SHR		AH,4
		ADD		AX,3030H	;转换成对应的SCII码
		XCHG	AH,AL
		MOV		WORD PTR HOUR,AX
							;保存到HOUR变量指示的前两个字节中
		MOV		AL,2		;2是“分”信息的偏移地址
		OUT		70H,AL
		JMP		$+2
		IN		AL,71H		;读取“分”信息
		MOV		AH,AL		
		AND		AL,0FH		
		SHR		AH,4
		ADD		AX,3030H	;转换成对应的SCII码
		XCHG	AH,AL
		MOV		WORD PTR MIN,AX		
							;保存到MIN变量指示的前两个字节中
		MOV		AL,0		;0是“秒”信息的偏移地址
		OUT		70H,AL
		JMP		$+2
		IN		AL,71H		;读取“秒”信息
		MOV		AH,AL		
		AND		AL,0FH		
		SHR		AH,4
		ADD		AX,3030H	;转换成对应的SCII码
		XCHG	AH,AL
		MOV		WORD PTR SEC,AX
							;保存到SEC变量指示的前两个字节中

			
CODE		ENDS
			END		BEGIN
			