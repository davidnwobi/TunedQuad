#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ezy_func_model.c"

#define PI 3.14159265358979323846

double integrand(double x, double B){
    return pow(x, 7) * sin(B * x);
}

int main(){
    printf("%f", integrate(integrand, 0, PI/2, 10000));
}