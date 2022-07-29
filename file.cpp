#include<bits/stdc++.h>
#include<stack>
using namespace std;

class stack{
    public:

    int *arr;
    int top;
    int size;

    stack(int size)
    {
        this -> size = size;
        arr = new int[size];
        int top = -1;
    }

    void push()
    {
        if(size - top > 1){
            top++;
            arr[top] = element;
        }
        else{
            cout<<"stack overflow"<<endl;
        }

    }
    void pop()
    {
        if(top >= 0){
            top--;
        }
        else{
            cout<<"stack is underflow"<<endl;
        }

    }
    int peek()
    {
        if(top >= 0){
            return arr[top];
        }
        else{
            cout<<"stack is empty"<<endl;
            return -1;
        }

    }
    bool isempty()
    {
        if(top == -1){
            return true;
        }
        else{
            return false;
        }

    }

};


















int main()
{




}

    /*stack <int> s;
    s.push(2);
    s.push(3);

    s.pop();
    cout<<"printing the top element in a stack"<<s.top()<<endl;
     
    if(s.empty()){
        cout<<"stack is empty"<<endl;

    } 
    else{
        cout<<"stack is non empty"<<endl;
    }/*



