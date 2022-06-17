extern int NbThreads;
extern int OnGPUFlag;
extern int MaxNbIters;
extern int NbIters;
extern T_real TOL;
extern T_real *data;
extern T_real *dataT;
extern T_real centroid[NbClusters][NbDims];
extern T_real package[NbClusters][NbDims];
extern int *label;
extern int count[NbClusters];

extern float Tms_init; 
extern float Tms_transpose;
extern float Tms_compute_assign;
extern float Tms_update;

extern double Ts_input;
extern double Ts_transfer;
extern double Ts_computation;
extern double Ts_transfer_computation;
extern double Ts_output;
extern double Ts_application;

extern double NumError;
extern FILE *fp;

void InputDataset(void);
void OutputResult(void);

void usage(int ExitCode, FILE *std); 
void CommandLineParsing(int argc, char *argv[]);

void PrintConfig(void);
void PrintResultsAndPerf(void);
