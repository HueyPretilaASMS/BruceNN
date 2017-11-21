using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using BruceNN.FrontEnd;

namespace Roxan
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            double Output = 0;
            if (EbolaCheck.IsChecked == true)
            {
                Output = BruceNN.FrontEnd.MiniNN.RunningInline6.runNeuralNetwork(1);
            } else
            {
                Output = BruceNN.FrontEnd.MiniNN.RunningInline6.runNeuralNetwork(0);
            }
            
            if (Output < 0.5)
            {
                DecisionLabel.Content = "I wouldn't treat you.";
                BruceNN.FrontEnd.MiniNN.RunningInline6.Speak("I wouldn't treat you. The supply in the simulation is limited, hence I have to prioritise patients.");
            } else
            {
                DecisionLabel.Content = "I would treat you.";
                BruceNN.FrontEnd.MiniNN.RunningInline6.Speak("I would treat you. The supply in the simulation is limited, hence I have to prioritise patients.");
            }
        }
    }
}
