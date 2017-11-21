using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.NeuralData;
using Encog.Persist;
using System;
using System.Threading;

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
                        /*/*BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new System.IO.FileInfo("network.txt"));
                        network.AddLayer(new BasicLayer(new ActivationStep(), true, 1));

                        double ageP, hasE, bloodT, fev, painI, painL, dia, exBlee, inBlee, deh, fa, exSw, loApp, head, sTh, men, gen;

                        #region Speech
                        using (SpeechSynthesizer synth = new SpeechSynthesizer())
                        {
                            #region Obtain Input
                            Console.Write("FirstName> ");
                            var fahs = Console.ReadLine();

                            Console.Write("contractedEbola> ");
                            var agePatient = Console.ReadLine();
                            Double.TryParse(agePatient, out ageP);

                            Console.Write("Age> ");
                            var hasEbola = Console.ReadLine();
                            Double.TryParse(hasEbola, out hasE);

                            Console.Write("Blood Type> ");
                            var bloodTy = Console.ReadLine();
                            Double.TryParse(bloodTy, out bloodT);

                            Console.Write("Fever> ");
                            var fever = Console.ReadLine();
                            Double.TryParse(fever, out fev);

                            Console.Write("Pain Intensity> ");
                            var painInt = Console.ReadLine();
                            Double.TryParse(painInt, out painI);

                            Console.Write("Pain Location> ");
                            var painLoc = Console.ReadLine();
                            Double.TryParse(painLoc, out painL);

                            Console.Write("Diarrhea> ");
                            var diarrhea = Console.ReadLine();
                            Double.TryParse(diarrhea, out dia);

                            Console.Write("External Bleeding> ");
                            var exBleeding = Console.ReadLine();
                            Double.TryParse(exBleeding, out exBlee);

                            Console.Write("Internal Bleeding> ");
                            var inBleeding = Console.ReadLine();
                            Double.TryParse(inBleeding, out inBlee);

                            Console.Write("Dehydration> ");
                            var Dehydration = Console.ReadLine();
                            Double.TryParse(Dehydration, out deh);

                            Console.Write("Fatigue> ");
                            var fat = Console.ReadLine();
                            Double.TryParse(fat, out fa);

                            Console.Write("Excess Sweating> ");
                            var exSweating = Console.ReadLine();
                            Double.TryParse(exSweating, out exSw);

                            Console.Write("Loss of Appetite> ");
                            var loAppetite = Console.ReadLine();
                            Double.TryParse(loAppetite, out loApp);

                            Console.Write("Headache> ");
                            var headAche = Console.ReadLine();
                            Double.TryParse(headAche, out head);

                            Console.Write("Sore Throat> ");
                            var sThroat = Console.ReadLine();
                            Double.TryParse(sThroat, out sTh);

                            Console.Write("Mental> ");
                            var mental = Console.ReadLine();
                            Double.TryParse(mental, out men);

                            Console.Write("Gender> ");
                            var gender = Console.ReadLine();
                            Double.TryParse(gender, out gen);

                            // Ebola | Age | BloodTy | Fever | PainInt | PainLoc | Diarrhea | HemmExt | HemmInt | Dehyd | Fatigue | ExSweat | Appetite | H/Ache | S/Throat | Mental | Gender
                            #endregion

                            synth.SelectVoice("Microsoft Zira Desktop");

                            synth.Speak("Thank you. I'll get to you shortly.");
                        }
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

                            #region Speech
                            using (SpeechSynthesizer synth = new SpeechSynthesizer())
                            {
                                // Output information about all of the installed voices. 
                                VoiceInfo info;
                                Console.WriteLine("Installed voices -");
                                foreach (InstalledVoice voice in synth.GetInstalledVoices())
                                {
                                    info = voice.VoiceInfo;
                                    Console.WriteLine(" Voice Name: " + info.Name);
                                }

                                synth.SelectVoice("Microsoft Zira Desktop");

                                if (output[0] == 1) { 
                                    synth.Speak("I can treat you.");
                                } else
                                {
                                    synth.Speak("I can't treat you.");
                                }
                            }
                            #endregion
                        } */
#endregion
                    }
                    else if (cmd == "TrainGen"){
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
