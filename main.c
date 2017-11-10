#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>



#define d 5
#define N 20
#define ni 1
double inner_prod(double a[],double b[])
{
    int i;
    double sum=0;   
    for(i=0;i<d;i++)
        sum+= a[i]*b[i];
    return sum;
}
int sgn(double x)
{
    if(x>=0)
       return 1;
    else
       return -1;
}
void print_mat(double A[][d])
{
    int i,j;
    for(i=0;i<N;i++)
        for(j=0;j<d;j++)
         {
           if(j<d-1)
               printf("%f\t",A[i][j]);
           else
               printf("%f\n",A[i][j]);
         }
}
double inner_prod_mat_vec(double Ai[][d],double xi[],int j)
{
    int k;
    double sum=0;
    for(k=0;k<d;k++)
        sum+=Ai[j][k]*xi[k];
    return sum;
}
void Gradient(double Ai[][d],double xi[],double g[],double z[],double ui[],double rho)
{
    int m,j,t;
    for(m=0;m<d;m++)
    {
        //Compute gradient.
                g[m] = rho*(xi[m]-z[m]+ui[m]);
                for(j=0;j<ni;j++)
                     g[m] -= Ai[j][m]/(1+exp(inner_prod_mat_vec(Ai,xi,j)));
    }               
}
double logistic_loss(double u)
{
    return log(1+exp(-u));
}
double Objective_i(double Ai[][d],double xi[],double z[], double ui[],double rho)
{
    int j,m;
    double sum=0;
    double pr;
    for(m=0;m<d;m++)
        sum += rho/2 * pow(xi[m]-z[m]+ui[m],2);
    for(j=0;j<ni;j++)
    {
        pr = inner_prod_mat_vec(Ai,xi, j);
        sum += logistic_loss(pr);
    }
    return sum;
}

double SSE(double a[],int n)
{
    int i;
    double sum=0;
    for(i=0;i<n;i++)
       sum+=pow(a[i],2);
    return sqrt(sum);
}
void GD(double xi[],double Ai[][d],double rho,double z[],double ui[])
{
    double g[d];
    int m,j,t=1;
    double epsilon=0.1;
    double sqgrad = 10;
    double gamma;
    double obj;
    while(sqgrad>epsilon && t<100)
    {
        Gradient(Ai,xi,g,z,ui,rho);
        gamma = (double)1/(t+2);
        for(m=0;m<d;m++)
            xi[m]-=gamma*g[m];
        obj = Objective_i(Ai,xi, z ,ui, rho);
        sqgrad = SSE(g,d);
        printf("Obj is: %f %0.11f\n",obj,sqgrad); 
        t++;
    }
}
void initialize(double A[][d])
{
    srand(1993);
    int i ,j;
    int qi;
    double x[d]={0.0};
    double pr, pi[d];
    pi[d-1] = 0;


    for(j=0;j<d-1;j++)
    {
        pr = (double)rand()/RAND_MAX;
        if(pr<0.5)
            x[j] = ((double)rand()-RAND_MAX/2)/RAND_MAX;
      //  printf("%f\n",x[j]);
            
    }
    for(i=0; i<N;i++)
    {
        for(j=0;j<d-1;j++)
             pi[j] = ((double) rand()-RAND_MAX/2)/RAND_MAX;
         qi = sgn(inner_prod(pi, x));
         A[i][d-1] = -qi;
         for(j=0;j<d-1;j++)
             A[i][j] = pi[j]*qi;
    }         
}
int main(int argc, char *argv[])
{
    double (*A) [d];
    A = malloc(sizeof(*A) * N);
    double rho = 1, lambda =1; 
    double Ai[ni][d];
    double z[d]={0};
    double ui[d]={0};
    double xi[d]={0};


    double g[d];


    int sendcount, recvcount, source, root,rank,numtasks;


    initialize(A);
//    print_mat(A); 

   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 
   if(numtasks*ni==N)
   { 
       sendcount = ni*d;
       recvcount = ni*d;
       source =1;
       MPI_Scatter(A,sendcount,MPI_DOUBLE,Ai,recvcount,
           MPI_DOUBLE,source,MPI_COMM_WORLD);
      
       if(rank==3)
           GD(xi,Ai,rho,z,ui);

   }
   else
       printf("You shoud specify %d processes.\n",N/ni);
   MPI_Finalize();
}
