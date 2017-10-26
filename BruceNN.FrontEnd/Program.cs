using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BruceNN.FrontEnd
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true) { 
                Console.Write(">");
                var cmd = Console.ReadLine();
                MiniNN.EthicalEngine.Init(cmd);
            }
        }
    }
}
