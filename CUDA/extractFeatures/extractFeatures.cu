 // includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <queue>

// includes for CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <sndfile.h>

// Complex data type
typedef float2 Complex;

////////////////////////////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////////////////////////////
void extractFeatures(int argc, char **argv, const char* filename);
float absComplex(Complex n);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
  if(argc == 2){
    std::string filename(argv[1]);
    filename += ".wav";
    extractFeatures(argc, argv, filename.c_str());
  }
  else if(argc == 3){
    int numFiles = atoi(argv[2]);
    std::string filename(argv[1]);
    std::stringstream ss;
    for(int i = 1; i <= numFiles; i++){
      ss << filename << i << ".wav";
//      std::cout << ss.str() << std::endl;
      extractFeatures(argc, argv, ss.str().c_str());
      ss.str("");
    }
  }
  else{
    std::cout << "Usage for single file: ./extractFeatures filename" << std::endl;
    std::cout << "or" << std::endl;
    std::cout << "Usage for multiple files: ./extractFeatures folder/filename numFiles" << std::endl;
    std::cout << "Note: folder/filename should not include spaces (' ') or the file extension ('.wav')." << std::endl;
  }
  exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// Extract Features
////////////////////////////////////////////////////////////////////////////////
void extractFeatures(int argc, char **argv, const char* filename){
    SNDFILE *waveFile;
    SF_INFO info;
    int signal_size, num_items;
    int *wave;
    int f,sr,c;

    // Open the WAV file.
    info.format = 0;
    waveFile = sf_open(filename,SFM_READ,&info);
    if (waveFile == NULL){
        printf("Failed to open the file.\n");
        exit(-1);
    }
    // Print some of the info, and figure out how much data to read.
    f = info.frames;
    sr = info.samplerate;
    c = info.channels;
    if(c != 1){
      printf("Error: file %s is not a mono wav file.\n", filename);
      exit(-1);
    }
    num_items = f*c;
    // Allocate space for the data to be read, then read it.
    wave = (int *) malloc(num_items*sizeof(int));
    signal_size = sf_read_int(waveFile,wave,num_items);
    sf_close(waveFile);

    // Allocate host memory for the signal
    int mem_size = sizeof(Complex) * signal_size;
    Complex *h_signal = (Complex *)malloc(mem_size);


    Complex val;
    for(int k = 0; k < signal_size; k++){
      val.x = wave[k];
      val.y = 0;
      h_signal[k] = val;
    }

    // CUDA FFT
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

    checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
                           cudaMemcpyDeviceToHost));

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    //h_signal is fft of wave
    //these variables are used for FFT of wave signal, only need half of FFT as it is mirrored
    float fft_signal_size, signal_time, frame_length, num_frames;
    int frame_size;
    fft_signal_size = signal_size/2;
    float freq = static_cast<float>(sr)/static_cast<float>(signal_size); // frequency per index of array
    signal_time = static_cast<float>(signal_size) / static_cast<float>(sr);
    frame_length = 0.002;
    num_frames = signal_time/(2*frame_length);
    num_frames = static_cast<int>(num_frames)+1;
    frame_size = (int)(signal_size/(2*num_frames));
    /*
    printf("\n");
    printf("signal size: %d\n", signal_size);
    printf("sample rate: %d\n", sr);
    printf("signal time: %f\n", signal_time);
    printf("num frames: %f\n", num_frames);
    printf("frame size: %d\n", frame_size);
    */
    float SCsum = 0.0;
    float BWsum = 0.0;
    float SRFsum = 0.0;
    float SFsum = 0.0;
    float sc;
    float bw;
    float srf;
    float sf;
    float maxFFT = 0.0;
    int h = frame_size/4;
    float th = 0.92;
    Complex fsumt; // fft sum total
    Complex fsumth;// fft sum threshold
    Complex fsumh; // fft sum h

    // calculate SC, BW, SRF, SF
    float num, den, gm, am;
    for(int i = 0; i < num_frames; i++){ // this will cycle through "frames"
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
      for(int j = begin; j < end && j < fft_signal_size; j++){ // this for loop cycles through values of a frame
        float absfft = absComplex(h_signal[j]);

        if(absfft > maxFFT){
          maxFFT = absfft;
        }

        num += (j-begin)*pow(absfft, 2);
        den += pow(absfft, 2);

        fsumt.x += h_signal[j].x;
        fsumt.y += h_signal[j].y;

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
      for(int j = begin; j < end && j < fft_signal_size; j++){ // this for loop cycles through values of a frame
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
    }

    // calculate min and max frequencies (at certain a certain threshold)
    float logFFT = 0;

    float maxf = 0;
    float minf = fft_signal_size;
    int s = 300/freq; // should only begin computing min/max at 300 Hz, ignore earlier values as they are most likely noise (should probably vary this value too)
    int threshold = -6;
    for(int i = s; i < fft_signal_size; i++){
      logFFT = 10*(log10(absComplex(h_signal[i])/maxFFT));
      if(logFFT > threshold && i < minf){
          minf = i;
      }
      if(logFFT > threshold && i > maxf){
        maxf = i;
      }
    }
    minf = minf*freq;
    maxf = maxf*freq;

    // calculate zero crossing rate
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

    // finalize feature values
    float SRFavg = SRFsum/num_frames;
    float SFavg = SFsum/num_frames;

    //report values
//    std::cout << "Spectral Centroid Sum: " << SCsum << std::endl;
//    std::cout << "Bandwidth: " << BWsum << std::endl;
//    std::cout << "SRF avg: " << SRFavg << std::endl;
//    std::cout << "SF avg: " << SFavg << std::endl;
//    std::cout << "ZCR: " << zcr << std::endl;
//    std::cout << "Min. Freq. at " << threshold << " dB: " << minf << std::endl;
//    std::cout << "Max. Freq. at " << threshold << " dB: " << maxf << std::endl;
//    std::cout << SCsum <<","<< BWsum <<","<< SRFavg <<","<< SFavg <<","<< zcr <<","<< minf <<","<< maxf << std::endl;
//    printf("\n");

    // output data
    if(argc == 2){
      std::string trainingname(argv[1]);
      trainingname += "Training.csv";
      std::ofstream datafile;
      datafile.open(trainingname.c_str(), std::ofstream::app);
      datafile << SCsum <<","<< BWsum <<","<< SRFavg <<","<< SFavg <<","<< zcr <<","<< minf <<","<< maxf <<"\n";
      datafile.close();

      // create the label
      std::string labelname(argv[1]);
      labelname += "Label.csv";
      std::ofstream labelfile;
      labelfile.open(labelname.c_str(), std::ofstream::app);
      std::string name = filename;
      if(name.find("AC") != std::string::npos){
        labelfile << 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("AK") != std::string::npos){
        labelfile << 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("AYW") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("BJ") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("CG") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("GC") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("GWT") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<"\n";
      }
      else if(name.find("NC") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<"\n";
      }
      labelfile.close();
    }
    else{
      std::ofstream datafile;
      datafile.open("TrainingData.csv", std::ofstream::app);
      datafile << SCsum <<","<< BWsum <<","<< SRFavg <<","<< SFavg <<","<< zcr <<","<< minf <<","<< maxf <<"\n";
      datafile.close();

      // create the label
      std::ofstream labelfile;
      labelfile.open("TrainingLabels.csv", std::ofstream::app);
      std::string name = filename;
      if(name.find("AC") != std::string::npos){
        labelfile << 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("AK") != std::string::npos){
        labelfile << 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("AYW") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("BJ") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("CG") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("GC") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<","<< 0 <<"\n";
      }
      else if(name.find("GWT") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<","<< 0 <<"\n";
      }
      else if(name.find("NC") != std::string::npos){
        labelfile << 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 0 <<","<< 1 <<"\n";
      }
      labelfile.close();
    }

    // cleanup memory
    free(wave);
    free(h_signal);
    checkCudaErrors(cudaFree(d_signal));
}
////////////////////////////////////////////////////////////////////////////////
// absComplex
////////////////////////////////////////////////////////////////////////////////
float absComplex(Complex n){
  float ret = 0;
  ret = sqrt(pow(n.x, 2) + pow(n.y, 2));
  return ret;
}
