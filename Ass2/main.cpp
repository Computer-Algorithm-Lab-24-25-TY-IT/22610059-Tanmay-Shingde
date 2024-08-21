#include <iostream>
#include <vector>
#include <climits>

using namespace std;

int matrixChainOrder(const vector<int>& p) {
    int n = p.size() - 1;
    vector<vector<int>> minCost(n, vector<int>(n, 0));

    for (int length = 2; length <= n; length++) {
        for (int i = 0; i < n - length + 1; i++) {
            int j = i + length - 1;
            minCost[i][j] = INT_MAX;
            for (int k = i; k < j; k++) {
                int q = minCost[i][k] + minCost[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
                if (q < minCost[i][j]) {
                    minCost[i][j] = q;
                }
            }
        }
    }

    return minCost[0][n - 1];
}

int main() {
    int numMatrices;
    cout << "Enter the number of matrices: ";
    cin >> numMatrices;

    vector<int> dimensions(numMatrices + 1);
    cout << "Enter the dimensions of the matrices in order.\n";
    cout << "For matrix A1 (with dimensions p1 x p2), enter p1 and p2.\n";
    cout << "For matrix A2 (with dimensions p2 x p3), enter p3, and so on.\n\n";

    for (int i = 0; i <= numMatrices; i++) {
        if (i == 0) {
            cout << "Enter dimension p1: ";
        } else {
            cout << "Enter dimension p" << i + 1 << ": ";
        }
        cin >> dimensions[i];
    }

    int minMultiplications = matrixChainOrder(dimensions);
    cout << "\nThe minimum number of multiplications is " << minMultiplications << endl;

    return 0;
}
