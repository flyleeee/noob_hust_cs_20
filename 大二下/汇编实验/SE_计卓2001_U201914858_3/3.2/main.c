#include<stdio.h>
#include<string.h>

typedef struct status {
	char SAMID[12];
	int SDA;
	int SDB;
	int SDC;
	int SP;
}status;

char tip[50] = "plz enter your name and password:";
char BNAME[12] = "wangyibin";
char BPASS[12] = "123456789";
char INAME[12] = {0};
char IPASS[12] = {0};

status s1 = { "000000001", 418479, 423908, 341, 0 };
status s2 = { "000000002", 847931, -423908, -423908, 0 };
status s3 = { "000000003", 418931, -423908, 21, 0 };
status s4 = { "000000004", 0, 12700, 0, 0 };
int n = 4;
int max_count = 3;
status lowf[100] = {0};
status midf[100] = {0};
status highf[100] = {0};

void __stdcall judge(int a, int b, int c);

void move_high(int i, status s)
{
	highf[i] = s;
}
void move_mid(int i, status s)
{
	midf[i] = s;
}
void move_low(int i, status s)
{
	lowf[i] = s;
}

int main()
{
	status s[4] = { s1, s2, s3, s4 };
	int result, i;
	for (i = 1; i <= max_count; i++)
	{
		printf("%s", tip);
		scanf_s("%s", INAME, 12);
		scanf_s("%s", IPASS, 12);
		if (strcmp(INAME, BNAME))
			continue;
		if (strcmp(IPASS, BPASS))
			continue;
		break;
	}
	for(i = 1; i <= n; i++)
		judge(s[i].SDA, s[i].SDB, s[i].SDC);
		__asm mov result, eax;
		if (result > 0)
			move_high(i, s[i]);
		else if (result == 0)
			move_mid(i, s[i]);
		else
			move_low(i, s[i]);


	return 0;
}
