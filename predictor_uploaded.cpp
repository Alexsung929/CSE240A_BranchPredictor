//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <math.h>
#include "predictor.h"
#include <vector>
#include <iostream>
//
// TODO:Student Information
//
const char *studentName = "";
const char *studentID = "";
const char *email = "";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
                         "Tournament", "Custom"};

// define number of bits required for indexing the BHT here.
int bpType;            // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

//
// TODO: Add your own Branch Predictor data structures here
//
// gshare
int ghistoryBits_gshare = 15; // Tournament
uint32_t bht_entries;
uint8_t *bht_gshare;
uint64_t ghistory;


// tournament
int hbtBits_tournament = 13; // Tournament
uint16_t bht_entries_tounament;
uint16_t global_predictor_entries;
int ghistoryBits_tournament = 13; // Tournament
uint8_t   *bht_tournament;
uint8_t   *chooser_tournament;
uint8_t   *global_predictor_tournament;
uint16_t  *pht_tournament;
uint16_t   pc_lower_bits;
uint16_t   ghistory_lower_bits;


// custom
#define NUMBER_OF_WEIGHTS 12
#define G_BITS 266
#define ADDR_BITS 11
#define INDEX_BITS 11
#define PHT_BITS 0
#define THRESHOLD 12
#define COUNTER_BITS 4
#define IS_USE_DIFFERENT_HLEN false

uint32_t  ghistoryBits_custom = G_BITS;
uint8_t   weightN_cumtom = NUMBER_OF_WEIGHTS;
uint32_t  addrBits_custom = ADDR_BITS; 
uint16_t  phtBits_custom  = PHT_BITS;
uint16_t  indexBits_custom = INDEX_BITS;

uint8_t   threshold = THRESHOLD;
uint32_t  perceptrons_entries;
int8_t   *perceptrons_table;
uint16_t  pht_entries;
uint16_t *pht_table;
int       prediction;

int       sat_high = 1<<(COUNTER_BITS-1);
int       sat_low  = -((1<<(COUNTER_BITS-1))-1);

uint16_t  L[NUMBER_OF_WEIGHTS];
uint16_t  index_hashed[NUMBER_OF_WEIGHTS];
uint8_t   alpha = 2;
std::vector<uint8_t>   ghistory_custom(G_BITS,0);

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor


int abs(int pred){
    return pred>=0? pred:-pred;
}

// custom functions
void init_custom(){
    perceptrons_entries = 1<<indexBits_custom;
    perceptrons_table = (int8_t*)malloc(sizeof(int8_t)*perceptrons_entries*weightN_cumtom);
    for (int i=0; i<perceptrons_entries; i++){
        perceptrons_table[i*weightN_cumtom+0] = 1; //bias
        for (int j=1; j<weightN_cumtom; j++){
            perceptrons_table[i*weightN_cumtom+j] = 0; 
        }
    }

    pht_entries = 1<<phtBits_custom;
    pht_table = (uint16_t*)malloc(sizeof(uint16_t)*pht_entries);
    for (int i=0; i<pht_entries; i++){
        pht_table[i] = 0;
    }

    // L[0] = 0;
    for (int i=1; i<NUMBER_OF_WEIGHTS; i++){
        // L[i] = pow(alpha,i);
        index_hashed[i] = 0;
    }
    
    uint16_t L_ref[] = {0, 2, 4, 9, 12, 18, 31, 54, 114, 145, 266};
    //    uint16_t L_ref[] = {0, 2, 4, 9, 12, 18, 31, 54, 114, 145,  163, 198, 244, 266};
    for (int i = 0; i < sizeof(L_ref) / sizeof(L_ref[0]); i++) {
        L[i] = L_ref[i];
    }

    ghistory = 0;
}

void folding_ghistory(){
    int j=0, offset=0;
    int temp=0;

    for (int i=1; i<NUMBER_OF_WEIGHTS; i++){
        offset = 0;
        j=0;
        index_hashed[i] = 0;
        int choose_L = i;
        if (IS_USE_DIFFERENT_HLEN){
            if (i==2) choose_L = 8;
            if (i==4) choose_L = 9;
            if (i==6) choose_L = 10;
        }
        while(j<L[choose_L] && j<ghistoryBits_custom){
            index_hashed[choose_L] ^= ((ghistory_custom[j]<<(j-offset)));
            j++;
            if (j%indexBits_custom==0)
                offset += indexBits_custom;
        }
    }
}

template <typename T1, typename T2>
uint16_t folding_history(T1& history, T2& bit_lens){
    uint16_t result = 0;
    uint64_t temp = history;
    int i = bit_lens;

    while(i>indexBits_custom){
        result = result ^ (temp & ((1<<indexBits_custom)-1));
        temp >>= indexBits_custom;
        i -= indexBits_custom;
    }
    result ^= (temp & ((1<<i)-1));


    return result;
}

void hashing_history(uint16_t& f_addr, uint16_t& f_pht){
    for (int i=1; i<NUMBER_OF_WEIGHTS; i++){
        index_hashed[i] ^= (f_addr ^ f_pht);
    }
}

uint8_t custom_predict(uint64_t pc){
    uint64_t addr             = pc & ( (1<<addrBits_custom) - 1); 
    uint16_t index_pc2pt      = pc & ( (1<<indexBits_custom) - 1); 
    uint16_t index_pc2pht     = pc & ( (1<<phtBits_custom) - 1); 
    uint16_t phistory         = pht_table[index_pc2pht];

    uint16_t folded_addr     = folding_history(addr, addrBits_custom);
    uint16_t folded_phsitory = folding_history(phistory, phtBits_custom);

    folding_ghistory();
    hashing_history(folded_addr, folded_phsitory);

    prediction = 0; 
    int temp = ghistory;
    int x_sign;
    prediction += perceptrons_table[index_pc2pt*weightN_cumtom + 0]; //w0 = bias
    for (int j=1; j<weightN_cumtom; j++){
        x_sign = ((temp&1) == 1)? 1:-1;
        prediction += x_sign * perceptrons_table[index_hashed[j] * weightN_cumtom + j];
        temp = temp>>1;
    }
    if (prediction>=0) return TAKEN;
    else return NOTTAKEN;
}

void train_custom(uint64_t pc, uint8_t outcome){
    uint16_t index_pc2pt      = pc & ( (1<<indexBits_custom) - 1); 
    uint16_t index_pc2pht     = pc & ( (1<<phtBits_custom) - 1); 

    int temp = ghistory;
    int update_value;
    int x_sign;
    int outcome_sign = (outcome==TAKEN)? 1:-1;
    if ( abs(prediction)<threshold || outcome_sign * prediction <= 0){
        perceptrons_table[index_pc2pt*weightN_cumtom + 0] += outcome_sign; //bias
        for (int j=1; j<weightN_cumtom; j++){
            x_sign = ((temp&1) == 1)? 1:-1;
            update_value = x_sign * outcome_sign;
            if (update_value>0 && perceptrons_table[index_hashed[j] * weightN_cumtom + j] < sat_high)
                perceptrons_table[index_hashed[j] * weightN_cumtom + j] += update_value;
            else if (update_value<0 && perceptrons_table[index_hashed[j] * weightN_cumtom + j] > sat_low)
                perceptrons_table[index_hashed[j] * weightN_cumtom + j] += update_value;    
            // perceptrons_table[index_hashed[j] * weightN_cumtom + j] += x_sign * outcome_sign;
            temp = temp>>1;
        }
    }

    ghistory_custom.insert(ghistory_custom.begin(), outcome);
    ghistory_custom.pop_back();
    pht_table[index_pc2pht] = ((pht_table[index_pc2pht]<<1) | outcome) & (pht_entries - 1);
    ghistory = ((ghistory << 1) | outcome);
}


void init_predictor()
{
    switch (bpType)
    {
    case STATIC:
        break;
    case GSHARE:
        init_gshare();
        break;
    case TOURNAMENT:
        init_tournament();
        break;
    case CUSTOM:
        init_custom();
        break;
    default:
        break;
    }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint32_t make_prediction(uint32_t pc, uint32_t target, uint32_t direct)
{
  // Make a prediction based on the bpType
    switch (bpType)
    {
    case STATIC:
        return TAKEN;
    case GSHARE:
        return gshare_predict(pc);
    case TOURNAMENT:
        return tournment_predict(pc);
    case CUSTOM:
        return custom_predict(pc);
    default:
        break;
    }
  // If there is not a compatable bpType then return NOTTAKEN
    return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//

void train_predictor(uint32_t pc, uint32_t target, uint32_t outcome, uint32_t condition, uint32_t call, uint32_t ret, uint32_t direct)
{
  if (condition)
  {
    switch (bpType)
    {
    case STATIC:
    case GSHARE:
        return train_gshare(pc, outcome);
    case TOURNAMENT:
        return train_tournament(pc, outcome);
    case CUSTOM:
        return train_custom(pc, outcome);
    default:
        break;
    }
  }
}
