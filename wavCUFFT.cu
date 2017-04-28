 // includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

//#include <arrayfire.h>
//#include <af/defines.h>
//#include <af/seq.h>
#include <sndfile.h>

// Complex data type
typedef float2 Complex;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void extractFeatures(int argc, char **argv, const char* filename);
float absComplex(Complex n);
Complex complexMul(Complex a, Complex b);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
  if(argc == 3){
    int numFiles = atoi(argv[2]);
    std::string filename(argv[1]);
    std::cout << "numfiles " << numFiles << std::endl;
    for(int i = 1; i <= numFiles; i++){
      std::stringstream ss;
      ss << i;
      std::string tfilename = filename+ss.str()+".wav";
      std::cout << tfilename << std::endl;
      extractFeatures(argc, argv, tfilename.c_str());
    }
  }
  else{
    std::cout << "usage error" << std::endl;
  }
  exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// Extract Features
////////////////////////////////////////////////////////////////////////////////
void extractFeatures(int argc, char **argv, const char* filename){
    findCudaDevice(argc, (const char **)argv);

    SNDFILE *waveFile;
    SF_INFO info;
    int signal_size, num_items;
    int *wave;
    int f,sr,c;

    /* Open the WAV file. */
    info.format = 0;
    waveFile = sf_open(filename,SFM_READ,&info);
    if (waveFile == NULL){
        printf("Failed to open the file.\n");
        exit(-1);
    }
    /* Print some of the info, and figure out how much data to read. */
    f = info.frames;
    sr = info.samplerate;
    c = info.channels;
//    printf("frames=%d\n",f);
//    printf("samplerate=%d\n",sr);
//    printf("channels=%d\n",c);
    num_items = f*c;
//    printf("num_items=%d\n",num_items);
    /* Allocate space for the data to be read, then read it. */
    wave = (int *) malloc(num_items*sizeof(int));
    signal_size = sf_read_int(waveFile,wave,num_items);
    sf_close(waveFile);
//    printf("Read %d items\n",signal_size);

    // Allocate host memory for the signal
    int mem_size = sizeof(Complex) * signal_size;
    Complex *h_signal = (Complex *)malloc(mem_size);

    Complex val;
    for(int k = 0; k < signal_size; k++){
      val.x = wave[k];
      val.y = 0;
      h_signal[k] = val;
    }

    // Allocate device memory for signal
    Complex *d_signal;
    checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size,
                               cudaMemcpyHostToDevice));

    // CUFFT plan simple API
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1));

    // Transform signal and kernel
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // Transform signal back
//    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

    checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
                           cudaMemcpyDeviceToHost));

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    /*
    int i, j;
    FILE *out;
    out = fopen("signal.txt","w");
    for (i = 0; i < signal_size; i += c){
      for (j = 0; j < c; ++j){
        fprintf(out,"%d ",wave[i+j]);
      }
      fprintf(out,"\n");
    }
    fclose(out);


    out = fopen("fftsignal.txt","w");
    for (i = 0; i < signal_size; i += c){
      for (j = 0; j < c; ++j){
        fprintf(out,"%f ",h_signal[i+j].x);
      }
      fprintf(out,"\n");
    }
    fclose(out);
    */

    //these variables are used for FFT of wave signal, only need half
    //as FFT
    float fft_signal_size, signal_time, frame_length, num_frames;
    int frame_size;
    fft_signal_size = signal_size/2;
    signal_time = static_cast<float>(signal_size) / static_cast<float>(sr);
    frame_length = 0.004; // 4 ms  // vary this value
    num_frames = signal_time/(2*frame_length);
    num_frames = static_cast<int>(num_frames)+1;
    frame_size = (int)(signal_size/(2*num_frames));
    //h_signal is fft of wave
    /*
    printf("\n");
    printf("signal size: %d\n", signal_size);
    printf("sample rate: %d\n", sr);
    printf("signal time: %f\n", signal_time);
    printf("num frames: %f\n", num_frames);
    printf("frame size: %d\n", frame_size);
    */
    int logFFT_size = sizeof(float) * (fft_signal_size);

    float* logFFT;
    logFFT = (float*)malloc(logFFT_size);
    float SCsum = 0.0;
    float BWsum = 0.0;
    float SRFsum = 0.0;
    float DSMsum = 0.0;
    float SFsum = 0.0;
//    float BERsum = 0.0;
    float sc;
    float bw;
    float srf;
    float dsm;
    float sf;
    float maxFFT = 0.0;
//    float ber;
    int h = frame_size/4;
    float th = 0.92;
    Complex fsumt; // fft sum total
    Complex fsumth;// fft sum threshold
    Complex fsumh; // fft sum h

    float num, den, gm, am;
    for(int i = 0; i < num_frames; i++){
      int begin = i*frame_size;
      int end = (i+1)*frame_size;

      sc = 0.0;
      srf = h;
      fsumt.x = 0.0;
      fsumt.y = 0.0;
      num = 0.0;
      den = 0.0;
      gm = 1.0;
      am = 0.0;
      for(int j = begin; j < end && j < fft_signal_size; j++){
        float absfft = absComplex(h_signal[j]);

        if(absfft > maxFFT){
          maxFFT = absfft;
        }

        num += (j-begin)*pow(absfft, 2);
        den += pow(absfft, 2);

        fsumt.x += h_signal[j].x;
        fsumt.y += h_signal[j].y;

        if((j < end-1) && (j < fft_signal_size-1)){
          dsm = fabs(absComplex(h_signal[j]) - absComplex(h_signal[j+1]));
          DSMsum += dsm;
        }

        gm *= pow(absfft, (1/frame_size));
        am += absfft;
      }

      sc = num/den;
      SCsum += sc;

      am = am/frame_size;
      sf = 10*log10(gm/am);
      SFsum  += sf;

      fsumth.x = fsumt.x*th;
      fsumth.y = fsumt.y*th;
      fsumh.x = 0.0;
      fsumh.y = 0.0;
      int sumflag = 1;

      bw = 0.0;
      num = 0.0;
      den = 0.0;
      for(int j = begin; j < end && j < fft_signal_size; j++){
        float absfft = absComplex(h_signal[j]);
        num += pow((j-begin-sc), 2)*pow(absfft, 2);
        den += pow(absfft, 2);

        if(sumflag){
          if(fsumh.x < fsumth.x){
            fsumh.x += h_signal[j].x;
            fsumh.y += h_signal[j].y;
          }
          else{
            sumflag = 0;
            if(j-begin > srf){
              srf = j-begin;
            }
          }
        }
      }
      bw = sqrt(num/den);
      BWsum += bw;

      SRFsum += srf;
/*
      Complex numer = fsumh;
      Complex denom = fsumt;
      Complex denomc;
      denomc.x = denom.x;
      denomc.y = -denom.y;

      Complex tnum = complexMul(numer, denomc);
      Complex tden = complexMul(denom, denomc);
      tnum.x = tnum.x/tden.x;
      tnum.y = tnum.y/tden.x;*/
/*      ber = absComplex(fsumh)/absComplex(fsumt);
      BERsum += ber;
      std::cout << "ber: " << ber << std::endl;*/

    }

    float maxf = 0;
    float minf = fft_signal_size;
    int threshold = -15; // vary this value
    for(int i = 0; i < fft_signal_size; i++){
      logFFT[i] = 10*(log10(absComplex(h_signal[i])/maxFFT));
      if(logFFT[i] > threshold && i < minf){
          minf = i;
      }
      if(logFFT[i] > threshold && i > maxf){
        maxf = i;
      }
    }
    float freq = static_cast<float>(sr)/static_cast<float>(signal_size);
    minf = minf*freq;
    maxf = maxf*freq;

    float SRFavg = SRFsum/num_frames;
    float DSMavg = DSMsum/num_frames;
    float SFavg = SFsum/num_frames;

    float zc = 0.0;
    for(int i = 0; i < signal_size-1; i++){
      if(wave[i] > 0 && wave[i+1] < 0){
        zc++;
      }
      else if(wave[i] < 0 && wave[i+1] > 0){
        zc++;
      }
    }
    float zcr = zc/signal_size;



    std::cout << "Spectral Centroid Sum: " << SCsum << std::endl;
    std::cout << "Bandwidth: " << BWsum << std::endl;
    std::cout << "SRF avg: " << SRFavg << std::endl;
//    std::cout << "BER sum: " << BERsum << std::endl;
    std::cout << "DSM avg: " << DSMavg << std::endl;
    std::cout << "SF avg: " << SFavg << std::endl;
    std::cout << "ZCR: " << zcr << std::endl;
    std::cout << "Min. Freq. at " << threshold << " dB: " << minf << std::endl;
    std::cout << "Min. Freq. at " << threshold << " dB: " << maxf << std::endl;
    printf("\n");

    // cleanup memory
    free(wave);
    free(h_signal);
    checkCudaErrors(cudaFree(d_signal));
}

float absComplex(Complex n){
  float ret = 0;
  ret = sqrt(pow(n.x, 2) + pow(n.y, 2));
  return ret;
}

Complex complexMul(Complex a, Complex b){
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
