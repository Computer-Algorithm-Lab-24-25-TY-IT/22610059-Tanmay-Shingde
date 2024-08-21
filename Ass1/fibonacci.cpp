#include <iostream>
using namespace std;

int fib(int n)
{
    if (n == 0 || n == 1)
    {
        return n;
    }

    int ans1 = fib(n - 1);
    int ans2 = fib(n - 2);

    return ans1 + ans2;
}

int main()
{
    int n;
    n = 32;

    int ans = fib(n);

    cout << ans << endl;
    return 0;
}