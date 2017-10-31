using Encog.ML.Data;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.NeuralData;
using Encog.Persist;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BruceNN.FrontEnd
{
    class Program
    {
        public static bool doStop = false, doSave = false, acceptInput = true;

        static void Main(string[] args)
        {
            while (true) {
                if (acceptInput == true) {
                    Console.Write(">");
                    var cmd = Console.ReadLine();

                    if (cmd == "Stop") {
                        doStop = true;
                    } else if (cmd == "Save"){
                        EncogDirectoryPersistence.SaveObject(new System.IO.FileInfo("network.txt"), MiniNN.EthicalEngine.NN);
                    } else if (cmd == "Load"){
                        #region Loading the old NN
                        BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new System.IO.FileInfo("network.txt"));

                        #region Obtain Input
                        Console.Write("FirstName> ");
                        var fahs = Console.ReadLine();

                        Console.Write("contractedEbola> ");
                        var agePatient = Console.ReadLine();
                        double ageP;
                        Double.TryParse(agePatient, out ageP);

                        Console.Write("Age> ");
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
#endregion

                        double[][] input = {
            new double[17] { ageP, hasE, bloodT, fev, painI, painL, dia, exBlee, inBlee, deh, fa, exSw, loApp, head, sTh, men, gen} };

                        INeuralDataSet inDat = new BasicNeuralDataSet(input, new double[][] { new double[0] });
                        foreach (IMLDataPair pair2 in inDat)
                        {
                            IMLData output = network.Compute(pair2.Input);
                            Console.WriteLine(pair2.Input[0] + @"," + pair2.Input[1]
                                              + @", answer=" + output[0]);
                            Console.WriteLine("provideCure> " + output[0]);
                        }
#endregion
                    } else if (cmd == "TrainGen"){
                        #region Generic Algorithm entrypoint w/ Multithread
                            if (acceptInput == true)
                            {
                                Thread trd = new Thread(new ThreadStart(MiniNN.EthicalEngine.InitGen))
                                {
                                    IsBackground = true
                                };
                                trd.Start();
                            }
                        #endregion
                    } else if (cmd == "TrainBac") {
                        #region Backpropagation entrypoint w/ Multithread
                            if (acceptInput == true)
                            {
                                Thread trd = new Thread(new ThreadStart(MiniNN.EthicalEngine.InitBac))
                                {
                                    IsBackground = true
                                };
                                trd.Start();
                            }
                        #endregion
                    }
                }
            }
        }
    }
}
