# coding=utf-8


def test():

    print('this is a Python program with PHP,')
    print("22,")
    print("21,")
    print("26,")
    print("22,")
    print("21,")
    print('Write a program that prints the numbers from 1 to 100. ')
    print('要求：打印1到100，遇到3的倍数，只打印“Fizz”,遇到5的倍数，打印“Buzz”,同时遇到3，5的倍数，打印“FizzBuzz”,')

    str = "中文"
    str = str.encode("utf-8")
    print(str)

    for x in range(1,101):
        print("Fizz"[x%3*4:]+"Buzz"[x%5*4:]or x)
    print(",")
    for x in range(1,11):
        print("qiangge_is_god"[x%3*14:]+"i_can't_believe_it"[x%5*18:] or x)
    print(",")
    Flag = True
    if Flag:
        print("Hello")
    else:
        print("World")
    print(',')



def Subtract(num1,num2):
    if num1 > num2:
        bignum = num1
        smallnum = num2
    else:
        bignum = num2
        smallnum = num1
    poor = bignum - smallnum
    print(poor)

    while poor != 0:
        if smallnum > poor:
            poor = smallnum - poor
            if poor == 0:
                print(poor)
                return 0
            else:
                poor = Subtract(poor, smallnum)

        else:
            poor = poor - smallnum
            print(poor)
            if poor == 0:
                print(poor)
                return 0
            else:
                poor = Subtract(poor, smallnum)


# num = input()
# num1 = int(num.split(',')[0])
# num2 = int(num.split(',')[1])
print(Subtract(20, 30))


# num1 = 100
# num2 = 50
# if num1 > num2:
#     num1 = num1 + num2
#     num2 = num1 - num2
#     num1 = num1 - num2
# print('num1:', num1, '>>>num2:', num2)
