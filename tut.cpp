#include<bits/stdc++.h>
#include<stack>
using namespace std;

class stack{
    public:

    int *arr;
    int top1;
    int top2;
    int size;


    twostack(int s)
    {
        this -> size = s;
        int top1 = -1;
        int top2 = s;
        arr = new arr[s];
    }


    void push1(int num)
    {
        if(top2 - top1 > 1){
            top1++;
            arr[top1] = num;
        }
        else{
            cout<<"stack is overflow";
        }
    }
    void push2(int num)
    {
        if(top2 - top1 > 1){
            top2--;
            arr[top2] = num;
        }
        else{
            cout<<"stack is overflow"<<endl;
        }
    }

    int pop1()
    {
        if(top1 >= 0){
            int ans = arr[top1];
            top1--;
            return ans;

        }
        else{
            return -1;
        }
    }
    int pop2()
    {
        if(top2 <= size){
            int ans = arr[top2];
            top2++;
            return ans;
        }
        else{
            return -1;
        }
    }







    

};








int main()
{

}