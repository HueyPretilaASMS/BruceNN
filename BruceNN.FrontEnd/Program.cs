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
        private Thread trd;

        static void Main(string[] args)
        {
            while (true) { 
                Console.Write(">");
                var cmd = Console.ReadLine();
                Thread trd = new Thread(new ThreadStart(MiniNN.EthicalEngine.InitGen))
                {
                    IsBackground = true
                };
                trd.Start();
            }
        }
    }
}
