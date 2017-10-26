
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Genetic;
using Encog.ML.Train;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.NeuralData;
using System;
using Encog.Neural.Networks.Training;
using Encog.ML;
using Encog.Neural.Networks.Training.Propagation.Back;

namespace BruceNN.FrontEnd.MiniNN
{
    class MLResettable { }

    public class SimulationScore : ICalculateScore
    {
        // Test subjects
        public static double[][] value = {
            new double[17] {1,30,0,1,9,1,0,1,1,1,1,0,0,1,0,1,1
}, new double[17] {1,88,1,1,9,3,1,0,0,1,0,0,0,1,0,1,1
}, new double[17] {1,41,2,0,3,1,0,1,1,1,1,1,1,0,0,0,1
}, new double[17] {1,21,0,1,6,1,1,1,0,1,1,1,0,1,0,0,0
}, new double[17] {0,47,2,0,3,3,1,1,0,0,0,0,0,0,0,0,0
}, new double[17] {0,38,3,0,9,2,0,1,1,1,0,0,0,1,0,0,1
}, new double[17] {1,60,1,0,8,2,0,0,0,0,1,0,1,0,0,1,1
}, new double[17] {1,42,1,0,8,2,0,1,1,0,1,1,0,1,0,1,1
}, new double[17] {1,9,0,0,6,4,0,0,0,1,0,0,0,0,1,1,1
}, new double[17] {0,86,3,1,6,2,0,0,1,1,0,1,1,0,0,1,1
}, new double[17] {0,19,0,1,2,4,1,1,1,1,0,1,0,1,0,1,0
}, new double[17] {1,100,3,0,8,3,0,1,1,1,1,0,1,0,0,1,1
}, new double[17] {1,15,1,0,6,2,1,1,1,1,1,0,1,0,1,1,1
}, new double[17] {0,87,2,0,9,4,1,1,1,1,0,1,0,1,1,0,1
}, new double[17] {1,54,3,1,7,4,0,0,1,0,1,0,0,0,1,1,0
}, new double[17] {0,48,3,0,6,2,0,1,0,1,1,1,1,0,0,1,1
}, new double[17] {0,56,1,0,8,1,0,0,0,1,1,0,0,0,0,0,1
}, new double[17] {1,16,2,0,2,3,1,1,1,1,0,1,1,0,1,0,1
}, new double[17] {0,7,3,1,10,1,1,1,1,1,1,0,0,0,1,1,1
}, new double[17] {0,55,2,1,4,3,1,0,1,1,1,0,1,0,1,0,1
}, new double[17] {1,83,3,0,5,1,0,1,0,1,0,1,0,0,1,1,0
}, new double[17] {0,45,3,1,1,4,0,1,1,0,0,1,1,1,0,0,1
}, new double[17] {1,12,0,0,1,3,1,1,0,1,1,1,0,1,1,0,1
}, new double[17] {1,28,2,0,2,1,0,1,0,1,0,0,1,0,1,0,0
}, new double[17] {0,76,2,0,9,1,0,0,1,0,1,1,1,0,1,0,1
}, new double[17] {1,91,1,0,3,4,0,1,1,1,0,0,0,0,0,0,0
}, new double[17] {1,52,1,0,8,2,1,1,1,0,0,1,0,1,0,1,0
}, new double[17] {0,74,3,1,2,1,0,1,0,0,0,0,0,0,1,0,0
}, new double[17] {0,99,1,0,3,2,1,0,1,1,1,1,0,1,1,0,1
}, new double[17] {1,95,3,1,6,2,1,1,1,0,0,0,1,0,1,0,0
}, new double[17] {0,40,1,0,4,2,0,1,1,1,0,1,1,0,0,1,0
}, new double[17] {0,64,3,0,8,2,0,1,1,0,1,0,1,1,1,1,1
}, new double[17] {0,34,1,0,3,2,1,0,0,1,0,1,0,1,1,0,0
}, new double[17] {0,46,2,1,6,2,0,1,1,1,1,1,1,1,1,1,1
}, new double[17] {0,51,1,0,4,2,1,1,1,1,0,0,0,1,0,1,0
}, new double[17] {1,79,0,1,2,3,0,0,0,1,0,1,0,1,0,1,0
}, new double[17] {0,68,0,0,3,1,0,0,1,0,0,0,1,0,0,1,0
}, new double[17] {1,85,3,0,8,4,0,0,0,1,1,1,1,1,1,0,1
}, new double[17] {0,69,1,0,7,4,0,1,1,1,0,0,1,0,1,0,0
}, new double[17] {1,59,1,1,8,3,1,1,1,1,0,1,0,1,1,0,0
}, new double[17] {0,2,0,0,10,4,1,1,1,1,1,0,0,1,1,0,1
}, new double[17] {0,80,3,1,10,4,0,0,1,1,1,1,0,1,0,1,0
}, new double[17] {1,31,2,1,5,2,0,1,0,1,1,0,1,1,0,0,1
}, new double[17] {1,62,0,1,6,2,0,0,0,0,0,1,1,0,1,0,0
}, new double[17] {1,22,2,0,5,2,0,0,1,0,1,0,0,1,0,0,0
}, new double[17] {0,90,0,0,4,2,1,1,1,0,1,1,1,0,1,1,1
}, new double[17] {0,81,3,1,9,3,1,1,1,0,0,0,0,0,0,1,1
}, new double[17] {0,70,3,0,3,4,1,1,0,0,0,0,1,1,0,0,0
}, new double[17] {1,5,3,1,2,3,1,1,0,1,1,1,1,0,0,0,1
}, new double[17] {0,93,1,0,3,4,1,0,1,1,0,1,1,0,0,1,1
}, new double[17] {1,44,2,1,5,1,1,0,1,0,0,0,1,0,1,0,0
}, new double[17] {1,17,3,1,4,3,1,1,1,1,1,1,0,0,1,0,1
}, new double[17] {0,1,0,0,5,4,1,1,1,0,1,0,0,0,0,0,0
}, new double[17] {1,35,0,1,9,1,1,0,1,1,1,0,0,1,0,0,0
}, new double[17] {0,4,3,0,3,2,1,1,0,1,0,1,0,0,1,1,0
}, new double[17] {1,39,3,0,8,4,0,1,0,0,0,0,1,0,0,1,1
}, new double[17] {1,72,2,1,3,3,1,0,0,1,1,1,1,0,0,0,1
}, new double[17] {1,92,0,0,5,2,1,0,0,0,0,1,1,1,1,0,1
}, new double[17] {1,14,3,0,4,1,0,1,0,0,0,0,0,1,0,0,1
}, new double[17] {0,89,1,1,5,3,0,1,0,0,0,0,0,0,0,0,1
}, new double[17] {0,10,1,1,3,2,0,0,0,0,0,0,0,0,0,0,1
}, new double[17] {1,97,0,0,4,4,0,1,0,0,1,0,0,0,0,1,1
}, new double[17] {1,43,3,0,6,4,0,0,1,1,0,0,0,0,0,1,0
}, new double[17] {1,23,3,1,7,1,1,0,1,1,0,0,1,0,0,1,1
}, new double[17] {1,26,1,0,7,3,1,1,0,0,0,1,0,0,0,0,0
}, new double[17] {0,63,2,0,1,1,0,1,1,0,0,0,1,1,0,1,1
}, new double[17] {0,84,3,0,7,2,1,0,0,1,0,1,0,0,0,0,1
}, new double[17] {0,73,2,1,4,3,0,0,1,1,1,1,1,0,0,0,0
}, new double[17] {0,37,2,0,8,2,1,1,0,1,0,0,0,0,0,1,0
}, new double[17] {0,20,2,1,6,4,0,0,0,0,1,0,0,1,1,1,0
}, new double[17] {0,6,0,1,9,1,0,1,0,0,0,1,0,0,1,1,0
}, new double[17] {0,78,0,1,8,3,0,0,1,0,1,1,1,0,1,1,0
}, new double[17] {0,27,1,1,7,4,1,0,1,0,1,1,0,0,0,0,0
}, new double[17] {0,98,3,1,6,1,1,0,0,1,1,1,0,0,0,0,1
}, new double[17] {1,82,1,0,5,2,1,1,0,0,0,0,0,0,0,0,0
}, new double[17] {1,57,2,0,6,2,1,0,1,1,0,1,0,0,1,0,1
}, new double[17] {1,25,3,0,7,2,0,1,0,1,0,0,0,1,0,1,1
}, new double[17] {1,3,3,1,9,1,0,1,1,0,1,0,1,1,1,1,0
}, new double[17] {1,65,0,1,2,4,0,0,0,0,0,1,0,1,1,0,0
}, new double[17] {0,13,2,0,8,4,0,0,0,1,0,0,1,0,0,1,1
}, new double[17] {0,77,1,0,9,1,1,0,0,0,1,1,0,1,1,0,0
}, new double[17] {1,71,3,0,8,4,1,1,1,0,1,1,1,0,0,1,0
}, new double[17] {0,11,2,1,5,2,0,1,1,1,1,0,0,0,1,1,1
}, new double[17] {1,32,2,1,1,3,1,1,0,1,1,0,1,0,0,1,1
}, new double[17] {0,53,2,0,7,1,1,1,1,0,0,0,1,0,1,1,1
}, new double[17] {0,66,2,1,1,2,0,0,0,1,1,0,0,1,0,0,0
}, new double[17] {0,58,3,1,9,4,0,0,0,1,0,1,1,1,1,0,0
}, new double[17] {0,29,3,0,1,1,1,1,1,1,0,1,1,0,0,0,0
}, new double[17] {0,18,2,1,6,2,1,1,0,1,1,1,0,1,1,0,1
}, new double[17] {1,36,3,1,1,3,0,0,0,1,0,1,1,0,0,1,1
}, new double[17] {0,33,0,0,4,2,1,1,1,0,0,1,0,1,0,0,1
}, new double[17] {0,8,3,0,5,1,0,1,0,1,1,1,1,0,1,1,1
}, new double[17] {1,50,0,1,4,3,0,0,1,1,0,0,1,0,1,1,0
}, new double[17] {0,49,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1
}, new double[17] {1,67,1,0,3,2,1,1,1,1,1,0,0,1,0,1,1
}, new double[17] {1,94,2,0,2,4,1,1,0,0,1,1,0,1,1,0,0
}, new double[17] {1,61,3,0,5,3,0,1,1,1,0,1,1,0,0,1,1
}, new double[17] {0,96,1,0,5,1,0,1,1,1,0,0,0,0,1,1,0
}, new double[17] {0,75,1,0,4,4,0,0,0,1,1,0,0,1,1,0,0
}, new double[17] {0,24,3,0,8,1,1,1,0,0,1,1,1,1,1,0,1}
        };

        public double CalculateScore(IMLMethod network)
        {
            int score = 0;
            // Simulation Loop
            foreach (double[] boop in value)
            {
                double[][] input = {boop};
                INeuralDataSet inDat = new BasicNeuralDataSet(input, value);
                foreach (IMLDataPair pair2 in inDat)
                {
                    IMLData output = EthicalEngine.NN.Compute(pair2.Input);
                    Console.WriteLine(pair2.Input[0] + @"," + pair2.Input[1]
                                      + @", answer=" + output[0]);
                    Console.WriteLine("provideCure> " + output[0]);

                    if ((int)output[0] == 0)
                    {
                        score -= 20;
                    } else
                    {
                        score += CalculatePoints(boop);
                    }
                }
            }
            return score;
        }

        public int CalculatePoints (double[] lol)
        {
            double[] lol1;

            // Age
            int agePoint = 0;
            if (lol[1] <= 1)
            {
                agePoint = 1;
            }
            else if (2 <= lol[1] && lol[1] <= 4)
            {
                agePoint = 2;
            }
            else if (5 <= lol[1] && lol[1] <= 10)
            {
                agePoint = 3;
            }
            else if (11 <= lol[1] && lol[1] <= 15)
            {
                agePoint = 5;
            }
            else if (16 <= lol[1])
            {
                agePoint = 4;
            }

            int painInt = 10 - (int)lol[3];

            lol1 = new double[] { lol[0], (double)agePoint, lol[2], (double)painInt, lol[4], lol[5], lol[6], lol[7], lol[8], lol[9], lol[10], lol[11], lol[12], lol[13], lol[14], lol[15], lol[16]};

            int total = 0;
            foreach (double lol2 in lol1)
            {
                total += (int)lol2;
            }

            return total;
        }

        public bool ShouldMinimize
        {
            get { return false; }
        }

        public bool RequireSingleThreaded
        {
            get { return false; }
        }
    }

    public class EthicalEngine
    {
        public static BasicNetwork NN;
        #region TRAINING VALUES
        // The CSV data is stored here
        // Ebola | Age | BloodTy | Fever | PainInt | PainLoc | Diarrhea | HemmExt | HemmInt | Dehyd | Fatigue | ExSweat | Appetite | H/Ache | S/Throat | Mental | Gender
        public static double[][] value = {
            new double[17] {1,30,0,1,9,1,0,1,1,1,1,0,0,1,0,1,1
}, new double[17] {1,88,1,1,9,3,1,0,0,1,0,0,0,1,0,1,1
}, new double[17] {1,41,2,0,3,1,0,1,1,1,1,1,1,0,0,0,1
}, new double[17] {1,21,0,1,6,1,1,1,0,1,1,1,0,1,0,0,0
}, new double[17] {0,47,2,0,3,3,1,1,0,0,0,0,0,0,0,0,0
}, new double[17] {0,38,3,0,9,2,0,1,1,1,0,0,0,1,0,0,1
}, new double[17] {1,60,1,0,8,2,0,0,0,0,1,0,1,0,0,1,1
}, new double[17] {1,42,1,0,8,2,0,1,1,0,1,1,0,1,0,1,1
}, new double[17] {1,9,0,0,6,4,0,0,0,1,0,0,0,0,1,1,1
}, new double[17] {0,86,3,1,6,2,0,0,1,1,0,1,1,0,0,1,1
}, new double[17] {0,19,0,1,2,4,1,1,1,1,0,1,0,1,0,1,0
}, new double[17] {1,100,3,0,8,3,0,1,1,1,1,0,1,0,0,1,1
}, new double[17] {1,15,1,0,6,2,1,1,1,1,1,0,1,0,1,1,1
}, new double[17] {0,87,2,0,9,4,1,1,1,1,0,1,0,1,1,0,1
}, new double[17] {1,54,3,1,7,4,0,0,1,0,1,0,0,0,1,1,0
}, new double[17] {0,48,3,0,6,2,0,1,0,1,1,1,1,0,0,1,1
}, new double[17] {0,56,1,0,8,1,0,0,0,1,1,0,0,0,0,0,1
}, new double[17] {1,16,2,0,2,3,1,1,1,1,0,1,1,0,1,0,1
}, new double[17] {0,7,3,1,10,1,1,1,1,1,1,0,0,0,1,1,1
}, new double[17] {0,55,2,1,4,3,1,0,1,1,1,0,1,0,1,0,1
}, new double[17] {1,83,3,0,5,1,0,1,0,1,0,1,0,0,1,1,0
}, new double[17] {0,45,3,1,1,4,0,1,1,0,0,1,1,1,0,0,1
}, new double[17] {1,12,0,0,1,3,1,1,0,1,1,1,0,1,1,0,1
}, new double[17] {1,28,2,0,2,1,0,1,0,1,0,0,1,0,1,0,0
}, new double[17] {0,76,2,0,9,1,0,0,1,0,1,1,1,0,1,0,1
}, new double[17] {1,91,1,0,3,4,0,1,1,1,0,0,0,0,0,0,0
}, new double[17] {1,52,1,0,8,2,1,1,1,0,0,1,0,1,0,1,0
}, new double[17] {0,74,3,1,2,1,0,1,0,0,0,0,0,0,1,0,0
}, new double[17] {0,99,1,0,3,2,1,0,1,1,1,1,0,1,1,0,1
}, new double[17] {1,95,3,1,6,2,1,1,1,0,0,0,1,0,1,0,0
}, new double[17] {0,40,1,0,4,2,0,1,1,1,0,1,1,0,0,1,0
}, new double[17] {0,64,3,0,8,2,0,1,1,0,1,0,1,1,1,1,1
}, new double[17] {0,34,1,0,3,2,1,0,0,1,0,1,0,1,1,0,0
}, new double[17] {0,46,2,1,6,2,0,1,1,1,1,1,1,1,1,1,1
}, new double[17] {0,51,1,0,4,2,1,1,1,1,0,0,0,1,0,1,0
}, new double[17] {1,79,0,1,2,3,0,0,0,1,0,1,0,1,0,1,0
}, new double[17] {0,68,0,0,3,1,0,0,1,0,0,0,1,0,0,1,0
}, new double[17] {1,85,3,0,8,4,0,0,0,1,1,1,1,1,1,0,1
}, new double[17] {0,69,1,0,7,4,0,1,1,1,0,0,1,0,1,0,0
}, new double[17] {1,59,1,1,8,3,1,1,1,1,0,1,0,1,1,0,0
}, new double[17] {0,2,0,0,10,4,1,1,1,1,1,0,0,1,1,0,1
}, new double[17] {0,80,3,1,10,4,0,0,1,1,1,1,0,1,0,1,0
}, new double[17] {1,31,2,1,5,2,0,1,0,1,1,0,1,1,0,0,1
}, new double[17] {1,62,0,1,6,2,0,0,0,0,0,1,1,0,1,0,0
}, new double[17] {1,22,2,0,5,2,0,0,1,0,1,0,0,1,0,0,0
}, new double[17] {0,90,0,0,4,2,1,1,1,0,1,1,1,0,1,1,1
}, new double[17] {0,81,3,1,9,3,1,1,1,0,0,0,0,0,0,1,1
}, new double[17] {0,70,3,0,3,4,1,1,0,0,0,0,1,1,0,0,0
}, new double[17] {1,5,3,1,2,3,1,1,0,1,1,1,1,0,0,0,1
}, new double[17] {0,93,1,0,3,4,1,0,1,1,0,1,1,0,0,1,1
}, new double[17] {1,44,2,1,5,1,1,0,1,0,0,0,1,0,1,0,0
}, new double[17] {1,17,3,1,4,3,1,1,1,1,1,1,0,0,1,0,1
}, new double[17] {0,1,0,0,5,4,1,1,1,0,1,0,0,0,0,0,0
}, new double[17] {1,35,0,1,9,1,1,0,1,1,1,0,0,1,0,0,0
}, new double[17] {0,4,3,0,3,2,1,1,0,1,0,1,0,0,1,1,0
}, new double[17] {1,39,3,0,8,4,0,1,0,0,0,0,1,0,0,1,1
}, new double[17] {1,72,2,1,3,3,1,0,0,1,1,1,1,0,0,0,1
}, new double[17] {1,92,0,0,5,2,1,0,0,0,0,1,1,1,1,0,1
}, new double[17] {1,14,3,0,4,1,0,1,0,0,0,0,0,1,0,0,1
}, new double[17] {0,89,1,1,5,3,0,1,0,0,0,0,0,0,0,0,1
}, new double[17] {0,10,1,1,3,2,0,0,0,0,0,0,0,0,0,0,1
}, new double[17] {1,97,0,0,4,4,0,1,0,0,1,0,0,0,0,1,1
}, new double[17] {1,43,3,0,6,4,0,0,1,1,0,0,0,0,0,1,0
}, new double[17] {1,23,3,1,7,1,1,0,1,1,0,0,1,0,0,1,1
}, new double[17] {1,26,1,0,7,3,1,1,0,0,0,1,0,0,0,0,0
}, new double[17] {0,63,2,0,1,1,0,1,1,0,0,0,1,1,0,1,1
}, new double[17] {0,84,3,0,7,2,1,0,0,1,0,1,0,0,0,0,1
}, new double[17] {0,73,2,1,4,3,0,0,1,1,1,1,1,0,0,0,0
}, new double[17] {0,37,2,0,8,2,1,1,0,1,0,0,0,0,0,1,0
}, new double[17] {0,20,2,1,6,4,0,0,0,0,1,0,0,1,1,1,0
}, new double[17] {0,6,0,1,9,1,0,1,0,0,0,1,0,0,1,1,0
}, new double[17] {0,78,0,1,8,3,0,0,1,0,1,1,1,0,1,1,0
}, new double[17] {0,27,1,1,7,4,1,0,1,0,1,1,0,0,0,0,0
}, new double[17] {0,98,3,1,6,1,1,0,0,1,1,1,0,0,0,0,1
}, new double[17] {1,82,1,0,5,2,1,1,0,0,0,0,0,0,0,0,0
}, new double[17] {1,57,2,0,6,2,1,0,1,1,0,1,0,0,1,0,1
}, new double[17] {1,25,3,0,7,2,0,1,0,1,0,0,0,1,0,1,1
}, new double[17] {1,3,3,1,9,1,0,1,1,0,1,0,1,1,1,1,0
}, new double[17] {1,65,0,1,2,4,0,0,0,0,0,1,0,1,1,0,0
}, new double[17] {0,13,2,0,8,4,0,0,0,1,0,0,1,0,0,1,1
}, new double[17] {0,77,1,0,9,1,1,0,0,0,1,1,0,1,1,0,0
}, new double[17] {1,71,3,0,8,4,1,1,1,0,1,1,1,0,0,1,0
}, new double[17] {0,11,2,1,5,2,0,1,1,1,1,0,0,0,1,1,1
}, new double[17] {1,32,2,1,1,3,1,1,0,1,1,0,1,0,0,1,1
}, new double[17] {0,53,2,0,7,1,1,1,1,0,0,0,1,0,1,1,1
}, new double[17] {0,66,2,1,1,2,0,0,0,1,1,0,0,1,0,0,0
}, new double[17] {0,58,3,1,9,4,0,0,0,1,0,1,1,1,1,0,0
}, new double[17] {0,29,3,0,1,1,1,1,1,1,0,1,1,0,0,0,0
}, new double[17] {0,18,2,1,6,2,1,1,0,1,1,1,0,1,1,0,1
}, new double[17] {1,36,3,1,1,3,0,0,0,1,0,1,1,0,0,1,1
}, new double[17] {0,33,0,0,4,2,1,1,1,0,0,1,0,1,0,0,1
}, new double[17] {0,8,3,0,5,1,0,1,0,1,1,1,1,0,1,1,1
}, new double[17] {1,50,0,1,4,3,0,0,1,1,0,0,1,0,1,1,0
}, new double[17] {0,49,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1
}, new double[17] {1,67,1,0,3,2,1,1,1,1,1,0,0,1,0,1,1
}, new double[17] {1,94,2,0,2,4,1,1,0,0,1,1,0,1,1,0,0
}, new double[17] {1,61,3,0,5,3,0,1,1,1,0,1,1,0,0,1,1
}, new double[17] {0,96,1,0,5,1,0,1,1,1,0,0,0,0,1,1,0
}, new double[17] {0,75,1,0,4,4,0,0,0,1,1,0,0,1,1,0,0
}, new double[17] {0,24,3,0,8,1,1,1,0,0,1,1,1,1,1,0,1}
        };

        public static double[][] value_Ans = {new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {1
}, new double[1] {0
}, new double[1] {1
}, new double[1] {1
}, new double[1] {1
}, new double[1] {0
}, new double[1] {0
}, new double[1] {0
}
        };
        #endregion

        public static void Init (string CSVFile)
        {
            INeuralDataSet trainingSet = new BasicNeuralDataSet(value, value_Ans);

            //TODO: Rewrite
            BasicNetwork network = new BasicNetwork();

            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 17));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 58));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 58));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 58));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 58));
            network.AddLayer(new BasicLayer(new ActivationStep(), true, 1));
            network.Structure.FinalizeStructure();
            network.Reset();
            
            IMLTrain trainMain = new MLMethodGeneticAlgorithm(() => {
                BasicNetwork result = network;
                ((IMLResettable)result).Reset();
                return result;
            }, new SimulationScore(), 10);

            //IMLTrain trainMain = new Backpropagation(network, trainingSet, 0.0001, 0.01);

            int epoch = 0;
            do
            {
                trainMain.Iteration();
                Console.WriteLine("Epoch #" + epoch + " Error:" + trainMain.Error);
                NN = network;
                epoch++;
            } while ((epoch < 50) && (trainMain.Error > 0.001));

            Console.WriteLine("Neural Network Results:");
            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);
                Console.WriteLine(pair.Input[0] + @"," + pair.Input[1]
                                  + @", actual=" + output[0] + @",ideal=" + pair.Ideal[0]);
            }

            while (true)
            {
                Console.Write("FirstName> ");
                var fahs = Console.ReadLine();

                Console.Write("Age> ");
                var agePatient = Console.ReadLine();
                double ageP;
                Double.TryParse(agePatient, out ageP);

                Console.Write("contractedEbola> ");
                var hasEbola = Console.ReadLine();
                double hasE;
                Double.TryParse(hasEbola, out hasE);

                Console.Write("Blood Type> ");
                var bloodTy = Console.ReadLine();
                double bloodT;
                Double.TryParse(bloodTy, out bloodT);

                Console.Write("Fever> ");
                var fever = Console.ReadLine();
                double fev;
                Double.TryParse(fever, out fev);
                
                Console.Write("Pain Intensity> ");
                var painInt = Console.ReadLine();
                double painI;
                Double.TryParse(painInt, out painI);
                
                Console.Write("Pain Location> ");
                var painLoc = Console.ReadLine();
                double painL;
                Double.TryParse(painLoc, out painL);

                Console.Write("Diarrhea> ");
                var diarrhea = Console.ReadLine();
                double dia;
                Double.TryParse(diarrhea, out dia);

                Console.Write("External Bleeding> ");
                var exBleeding = Console.ReadLine();
                double exBlee;
                Double.TryParse(exBleeding, out exBlee);

                Console.Write("Internal Bleeding> ");
                var inBleeding = Console.ReadLine();
                double inBlee;
                Double.TryParse(inBleeding, out inBlee);

                Console.Write("Dehydration> ");
                var Dehydration = Console.ReadLine();
                double deh;
                Double.TryParse(Dehydration, out deh);

                Console.Write("Fatigue> ");
                var fat = Console.ReadLine();
                double fa;
                Double.TryParse(fat, out fa);

                Console.Write("Excess Sweating> ");
                var exSweating = Console.ReadLine();
                double exSw;
                Double.TryParse(exSweating, out exSw);

                Console.Write("Loss of Appetite> ");
                var loAppetite = Console.ReadLine();
                double loApp;
                Double.TryParse(loAppetite, out loApp);

                Console.Write("Headache> ");
                var headAche = Console.ReadLine();
                double head;
                Double.TryParse(headAche, out head);

                Console.Write("Sore Throat> ");
                var sThroat = Console.ReadLine();
                double sTh;
                Double.TryParse(sThroat, out sTh);

                Console.Write("Mental> ");
                var mental = Console.ReadLine();
                double men;
                Double.TryParse(mental, out men);

                Console.Write("Gender> ");
                var gender = Console.ReadLine();
                double gen;
                Double.TryParse(gender, out gen);

                // Ebola | Age | BloodTy | Fever | PainInt | PainLoc | Diarrhea | HemmExt | HemmInt | Dehyd | Fatigue | ExSweat | Appetite | H/Ache | S/Throat | Mental | Gender

                double[][] input = {
            new double[17] { ageP, hasE, bloodT, fev, painI, painL, dia, exBlee, inBlee, deh, fa, exSw, loApp, head, sTh, men, gen} };

                INeuralDataSet inDat = new BasicNeuralDataSet(input, value_Ans);
                foreach (IMLDataPair pair2 in inDat)
                {
                    IMLData output = network.Compute(pair2.Input);
                    Console.WriteLine(pair2.Input[0] + @"," + pair2.Input[1]
                                      + @", answer=" + output[0]);
                    Console.WriteLine("provideCure> " + output[0]);
                }
            }
        }
    }
}
