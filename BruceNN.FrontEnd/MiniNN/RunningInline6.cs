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
using System.Linq;
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
                synth.Speak("I can treat you.");

                synth.Dispose();
            }
        }

        #region Separate Thread
        public static string YesOrNo;
        public static bool Listening = false;

        public static void Listen ()
        {
            Thread trd = new Thread(new ThreadStart(ListenThread))
            {
                IsBackground = true
            };
            trd.Start();

            Console.WriteLine("Press return to end recording");
            Console.ReadLine();
            Listening = false;
        }

        public static WaveIn waveSource = null;
        public static WaveFileWriter waveFile = null;

        static void ListenThread()
        {
            waveSource = new WaveIn();
            waveSource.WaveFormat = new WaveFormat(44100, 1);

            waveSource.DataAvailable += new EventHandler<WaveInEventArgs>(waveSource_DataAvailable);
            waveSource.RecordingStopped += new EventHandler<StoppedEventArgs>(waveSource_RecordingStopped);

            waveFile = new WaveFileWriter("dead.wav", waveSource.WaveFormat);

            waveSource.StartRecording();

            Listening = true;

            while (Listening == true)
            { }

            waveSource.StopRecording();
        }

        static void waveSource_DataAvailable(object sender, WaveInEventArgs e)
        {
            if (waveFile != null)
            {
                waveFile.Write(e.Buffer, 0, e.BytesRecorded);
                waveFile.Flush();
            }
        }

        static void waveSource_RecordingStopped(object sender, StoppedEventArgs e)
        {
            if (waveSource != null)
            {
                waveSource.Dispose();
                waveSource = null;
            }

            if (waveFile != null)
            {
                waveFile.Dispose();
                waveFile = null;
            }
        }
        #endregion
    }
}
