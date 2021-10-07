#ifndef DATATYPES_H
#define DATATYPES_H

typedef struct recv{
    double elem1;
    double elem2;
}recvtype;

typedef struct send{
    int n;
    int m;
    int mits;
    double alpha;
    double relax;
    double tol;
}sendtype;

#endif