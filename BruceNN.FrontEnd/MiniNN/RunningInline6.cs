using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.NeuralData;
using Encog.Persist;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Speech.Recognition;
using System.Speech.Synthesis;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BruceNN.FrontEnd.MiniNN
{
    public class RunningInline6
    {
        public static void runNeuralNetwork ()
        {
            #region Loading the old NN
            BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new System.IO.FileInfo("network.txt"));
            network.AddLayer(new BasicLayer(new ActivationStep(), true, 1));

            double ageP, hasE, bloodT, fev, painI, painL, dia, exBlee, inBlee, deh, fa, exSw, loApp, head, sTh, men, gen;

            #region Obtain Input
            Console.Write("FirstName> ");
            var fahs = Console.ReadLine();

            Console.Write("contractedEbola> ");
            var agePatient = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(agePatient, out ageP);

            Console.Write("Age> ");
            var hasEbola = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(hasEbola, out hasE);

            Console.Write("Blood Type> ");
            var bloodTy = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(bloodTy, out bloodT);

            Console.Write("Fever> ");
            var fever = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(fever, out fev);

            Console.Write("Pain Intensity> ");
            var painInt = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(painInt, out painI);

            Console.Write("Pain Location> ");
            var painLoc = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(painLoc, out painL);

            Console.Write("Diarrhea> ");
            var diarrhea = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(diarrhea, out dia);

            Console.Write("External Bleeding> ");
            var exBleeding = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(exBleeding, out exBlee);

            Console.Write("Internal Bleeding> ");
            var inBleeding = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(inBleeding, out inBlee);

            Console.Write("Dehydration> ");
            var Dehydration = Listen().ToString(); //Console.ReadLine();
            Double.TryParse(Dehydration, out deh);

            Console.Write("Fatigue> ");
            var fat = Listen().ToString(); //Console.ReadLine();
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
                if (output[0] == 1)
                {
                    Speak("I can treat you.");
                }
                else
                {
                    Speak("I can't treat you.");
                }
                #endregion
            }
            #endregion
        }

        public static void Speak (string Speech)
        {
            using (SpeechSynthesizer synth = new SpeechSynthesizer())
            {
                synth.SelectVoice("Microsoft Zira Desktop");
                synth.Speak(Speech);

                synth.Dispose();
            }
        }

        #region Separate Thread
        public static int YesOrNo;
		static bool completed, finished;
        public static bool Listening = false;

        public static int Listen ()
        {
            Listening = true;
            finished = false;
            Thread trd = new Thread(new ThreadStart(ListenThread))
            {
                IsBackground = true
            };
            trd.Start();
            ListenThread();

            Listening = false;
            return YesOrNo;
        }

        static void ListenThread()
        {
            /*using (SpeechRecognitionEngine listen = new SpeechRecognitionEngine())
            {
                listen.SetInputToDefaultAudioDevice();

                //Possible choices given then loaded into a grammarbuilder
                Choices inPos = new Choices();
                GrammarBuilder gramBuil = new GrammarBuilder();
                inPos.Add(new string[] { "yes", "no" });
                gramBuil.Append(inPos);

                //Load the grammar
                listen.LoadGrammar(new Grammar(gramBuil));

                listen.SpeechRecognized += new EventHandler<SpeechRecognizedEventArgs>(foundText);

                listen.Recognize();
            }*/

            using (SpeechRecognitionEngine recognizer = new SpeechRecognitionEngine())
            {
                // Create and load the exit grammar.
                Grammar exitGrammar = new Grammar(new GrammarBuilder("exit"));
                exitGrammar.Name = "Exit Grammar";
                recognizer.LoadGrammar(exitGrammar);

                // Create and load the dictation grammar.
                Grammar dictation = new DictationGrammar();
                dictation.Name = "Dictation Grammar";
                recognizer.LoadGrammar(dictation);

                // Attach event handlers to the recognizer.
                recognizer.SpeechRecognized +=
                  new EventHandler<SpeechRecognizedEventArgs>(
                    SpeechRecognizedHandler);
                recognizer.RecognizeCompleted +=
                  new EventHandler<RecognizeCompletedEventArgs>(
                    RecognizeCompletedHandler);

                // Assign input to the recognizer.
                recognizer.SetInputToDefaultAudioDevice();

                // Begin asynchronous recognition.
                Console.WriteLine("Starting recognition...");
                completed = false;
                recognizer.RecognizeAsync(RecognizeMode.Multiple);

                // Wait for recognition to finish.
                while (!completed)
                {
                    Thread.Sleep(333);
                }

                if (!finished)
                    ListenThread();
                Console.WriteLine("Done.");
            }
        }

		static void SpeechRecognizedHandler(object sender, SpeechRecognizedEventArgs e)
		{
		    Console.WriteLine("  Speech recognized:");
		    string grammarName = "<not available>";
		    if (e.Result.Grammar.Name != null && !e.Result.Grammar.Name.Equals(string.Empty))
		    {
			    grammarName = e.Result.Grammar.Name;
		    }
		    Console.WriteLine("    {0,-17} - {1}",
			grammarName, e.Result.Text);

		    if (grammarName.ToLower().Equals("yes"))
		    {
                YesOrNo = 1;
			    ((SpeechRecognitionEngine)sender).RecognizeAsyncCancel();
                Console.WriteLine("Yes.");
                completed = true;
                finished = true;
            } else if (grammarName.ToLower().Equals("no"))
            {
                YesOrNo = 0;
                Console.WriteLine("No.");
                ((SpeechRecognitionEngine)sender).RecognizeAsyncCancel();
                completed = true;
                finished = true;
            } else if (grammarName.ToLower().Equals("yeah"))
            {
                YesOrNo = 0;
                Console.WriteLine("Yes.");
                ((SpeechRecognitionEngine)sender).RecognizeAsyncCancel();
                completed = true;
                finished = true;
            }
        }

		static void RecognizeCompletedHandler(object sender, RecognizeCompletedEventArgs e)
		{
		      Console.WriteLine("  Recognition thread completed.");
		      completed = true;
		}
        #endregion
    }
}
