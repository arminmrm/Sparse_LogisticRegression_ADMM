#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <time.h>


#define d 29
#define N 100
#define ni 20
#define NP N/ni

double CLOCK(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return(t.tv_sec*1000)+(t.tv_nsec*1e-6);
}


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
    
    //Compute gradient.
    int m,j,t;
    for(m=0;m<d;m++)
    {
        g[m] = rho*(xi[m]-z[m]+ui[m]);
        for(j=0;j<ni;j++)
             g[m] -= Ai[j][m]/((1+exp(inner_prod_mat_vec(Ai,xi,j)))*N);
    }               
}


double logistic_loss(double u)
{
    return log(1+exp(-u))/N;
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


void GD(double xi[],double Ai[][d],double rho,double z[],double ui[],int rank)
{
    double g[d];
    int m,j,t=1;
    double epsilon=0.01;
    double sqgrad = 10;
    double gamma;
    double obj;
    while(sqgrad>epsilon && t<1000)
    {
        Gradient(Ai,xi,g,z,ui,rho);
        gamma = (double)1/(t+2);
        for(m=0;m<d;m++)
            xi[m]-=gamma*g[m];
        obj = Objective_i(Ai,xi, z ,ui, rho);
        if(rank==NP)
         printf("OBJ %f\t%f \n",obj,sqgrad);
        sqgrad = SSE(g,d);
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
            x[j] = 100*((double)rand()-RAND_MAX/2)/RAND_MAX;
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

void read_data(double A[][d],char *fname){
    int i,j;
    FILE *ptr_file;
    float val;
    ptr_file =fopen(fname, "r");
    for(i=0;i<N;i++)
        for(j=0;j<d;j++)
        {
            fscanf(ptr_file,"%f",&val);
            if(j==d-2)
                A[i][j]=val/1000;
            else
                A[i][j]=val;
        }
    fclose(ptr_file);
}


void add_arr(double a[],double b[],double c[])
{
    int i;
    for(i=0;i<d;i++)
        c[i] = a[i]+b[i];
}
void prox_op(double z[],double z_sum[],double par){
    int i;
    for(i=0;i<d-1;i++)
    {
       if(z_sum[i]>par)
           z[i] = z_sum[i]-par;
       else if(z_sum[i]<-par)
           z[i] = z_sum[i]+par;
       else
           z[i]=0;
    }
    z[d-1] = z_sum[d-1];
}
void scale(double a[],double alpha){
    int i;
    for(i=0;i<d;i++)
        a[i] = a[i]/alpha;
}
void adapt_ui(double ui[d], double xi[d], double z[d])
{
     int i;
     for(i=0;i<d;i++)
        ui[i]+=(xi[i]-z[i]);
}
void cp_arr(double z[],double z_prev[]){
    int i;
    for(i=0;i<d;i++)
        z_prev[i]=z[i];
}
double  resid(double xi[],double z[]){
    int i;
    double sum=0;
    for(i=0;i<d;i++)
        sum+=pow(xi[i]-z[i],2);
    return sum;
}    
double l1_norm(double a[]){
    int i;
    double l1=0;
    for(i=0;i<d;i++)
        l1+=abs(a[i]);
    return l1;
}
void write_var(double z[d],const char *file){
    FILE *ptr_file;
    int i;
    ptr_file =fopen(file, "w");
    for(i=0;i<d;i++)
        fprintf(ptr_file,"%f\n", z[i]);
    fclose(ptr_file);
}
int main(int argc, char *argv[])
{
    double (*A) [d];
    A = malloc(sizeof(*A) * N);
    double rho = 1, lambda =0.05; 
    double z[d]={0};
    double ui[d]={0};
    double xi[d]={0};
    double Ai[ni][d];
    double x_bar[d];
    double u_bar[d];
    double z_sum[d];
    double z_prev[d];
    double g[d];
    double ri=10,ri_2=10,OBJ_i;    
    int i;
    int sendcount, recvcount, root,rank,numtasks;
    double tstart,tend;
    //initialize(A);
    read_data(A,"dataset_c");
    printf("%f\n",A[0][d-2]);

    tstart = CLOCK();
//    print_mat(A); 

   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 
   if(numtasks==NP)
   { 
       sendcount = ni*d;
       recvcount = ni*d;
     
       MPI_Scatter(A,sendcount,MPI_DOUBLE,Ai,recvcount,
           MPI_DOUBLE,1,MPI_COMM_WORLD);
     for(i=0;i<400;i++){
       GD(xi,Ai,rho,z,ui,rank);
       ri = resid(xi,z);
//      printf("Rank %d and xi: %f %f %f %f %f\n",rank,xi[0],xi[1],xi[2],xi[3],xi[4]);
       OBJ_i = Objective_i(Ai,xi, z, ui,rho);
       MPI_Reduce (xi,x_bar,d,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
       MPI_Reduce (ui,u_bar,d,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
       MPI_Reduce (&ri,&ri,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
       MPI_Reduce (&OBJ_i,&OBJ_i,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
       if(rank==0)
       {
       //    printf("Rank %d and xi: %f %f %f %f %f\n",rank,x_bar[0],x_bar[1],x_bar[2],x_bar[3],x_bar[4]);
           cp_arr(z,z_prev);
           scale(x_bar,(double) NP);
           scale(u_bar,(double) NP);
       //    printf("Rank %d and x bar: %f %f %f %f %f\n",rank,x_bar[0],x_bar[1],x_bar[2],x_bar[3],x_bar[4]);
       //    printf("Rank %d and u bar: %f %f %f %f %f\n",rank,u_bar[0],u_bar[1],u_bar[2],u_bar[3],u_bar[4]);
           ri = sqrt(ri);
           add_arr(x_bar,u_bar,z_sum);
      //     printf("Rank %d and z sum: %f %f %f %f %f\n",rank,z_sum[0],z_sum[1],z_sum[2],z_sum[3],z_sum[4]);
           prox_op(z,z_sum,lambda/(rho*NP)); 
      //     printf("Rank %d and z sum: %f %f %f %f %f\n",rank,z[0],z[1],z[2],z[3],z[4]);
           ri_2 = rho*sqrt(NP)*sqrt(resid(z,z_prev));
           printf("residual 1: %f, residual 2: %f,%f\n",ri,ri_2,OBJ_i+lambda*l1_norm(z));
       }
       MPI_Barrier(MPI_COMM_WORLD);
       MPI_Bcast ( z, d, MPI_DOUBLE,0, MPI_COMM_WORLD );
      // printf("Rank %d and z sum: %f %f %f %f %f\n",rank,z[0],z[1],z[2],z[3],z[4]);
       MPI_Bcast ( &ri, 1, MPI_DOUBLE,0, MPI_COMM_WORLD );
       MPI_Bcast ( &ri_2, 1, MPI_DOUBLE,0, MPI_COMM_WORLD );
       adapt_ui( ui,  xi,  z);
//       printf("Rank %d and u after update: %f %f %f %f %f\n",rank,ui[0],ui[1],ui[2],ui[3],ui[4]);
    //
     }
   }
   else
       printf("You shoud specify %d processes.\n",NP);
   MPI_Finalize();
   tend = CLOCK();
   //printf("Time taken: %f\n",tend-tstart);
   if(rank==0){
       write_var(z,argv[1]);
       printf("Time taken: %f\n",tend-tstart);
   }
    return 0;
}
