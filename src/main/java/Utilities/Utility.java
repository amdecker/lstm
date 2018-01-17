package main.java.Utilities;

import java.awt.*;
import java.io.*;
import java.util.Scanner;


/**
 * Provides shortcuts for commonly used tasks
 *
 * @author Amos Decker
 * @version 11/8/17
 */
public class Utility
{
    /**
     * Prints out one character at a time with a pause between each character
     *
     * @param text     the text to be printed
     * @param duration amount of time between each character in miliseconds
     */
    public static void sleepy(String text, int duration)
    {
        for (int c = 0; c < text.length(); c++)
        {
            print("" + text.charAt(c));
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
        }
    }

    /**
     * Does a little rotation by printing over what has been printed
     * @param numTimes number of times through a full rotation
     * @param duration the duration of the pause between each movement, anything below 300 seems to mess it up
     */
    public static void loadingRotation(int numTimes, int duration)
    {
        for (int i = 0; i < numTimes; i++)
        {
            print("\\");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r\r");
            print("|");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");
            print("/");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");
            print("-");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");
            print("\\");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r\r");
            print("|");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");
            print("/");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");
            print("-");
            try
            {
                Thread.sleep(duration);
            } catch (Exception e) {}
            print("\r");

        }
    }
    
    // MOST System.out.print() variations
    /**
     * Shorter way to type System.out.print()
     *
     * @param toPrint
     */
    public static void print(String toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(int toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(double toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(long toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(char toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(float toPrint)
    {
        System.out.print(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void print(Object toPrint)
    {
        System.out.print(toPrint);
    }


    // MOST System.out.println() variations
    /**
     * Shorter way to type System.out.println() for strings
     *
     * @param toPrint
     */
    public static void println(String toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.println() for ints
     *
     * @param toPrint
     */
    public static void println(int toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void println(double toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void println(long toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void println(float toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void println(Object toPrint)
    {
        System.out.println(toPrint);
    }

    /**
     * Shorter way to type System.out.print() for ints
     *
     * @param toPrint
     */
    public static void println(char toPrint)
    {
        System.out.println(toPrint);
    }

    // Read text from file
    /**
     * @param fileName name of the file to read
     * @return entireFile which is a string with all the words (and spaces and \\n) or "false" if the file is not found
     */
    public static String readFile(String fileName)
    {
        try
        {
            File fileObj = new File(fileName);
            Scanner sc = new Scanner(fileObj);
            String entireFile = "";
            String line;

            // keep going through the file until the end of the file is reached
            while (sc.hasNextLine())
            {
                entireFile += sc.nextLine() + '\n';
            }
            return entireFile;
        }
        catch (FileNotFoundException e)
        {
            return "*******false*********";
        }
    }

    /**
     * writes to a file
     * @param filepath
     * @param content
     * @param append
     * @throws IOException
     */
    public static void writeFile(String filepath, String content, boolean append) throws IOException
    {
        FileWriter file = new FileWriter(filepath, append);
        file.write(content);
        file.close();
    }

    // System Command
    /**
     * Runs a system command
     *
     * code modified from https://alvinalexander.com/java/edu/pj/pj010016
     *
     * @param commandArray: the command to be run, an array of the args like {"ls", "/User/Desktop"}
     * @return the output of the command or prints the error there is an error in your command
     */
    public static StringBuilder exec(String[] commandArray, String directory)
    {
        String s = null;
        StringBuilder output = new StringBuilder("");

        try {
            if (directory == null)
            {
                directory =
                        System.getProperty("user.dir");
            }
            // using the Runtime exec method:
            //Process p = Runtime.getRuntime().exec(commandArray);
            ProcessBuilder pb = new ProcessBuilder(commandArray);
            //String[] commands = {"commands"};
            //ProcessBuilder builder = new ProcessBuilder(commands);

            pb.directory(new File(directory));
            Process p = pb.start();


            BufferedReader stdInput = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            // read the output from the command
            while ((s = stdInput.readLine()) != null) {
                output.append(s);
                output.append('\n');

            }
            // remove the last \n
            if(output.length() > 0)
            {
                output.deleteCharAt(output.length() - 1);
            }

            // read any errors from the attempted command
            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }

            //System.exit(0); // this was here but it stops the entire thing

        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }
        return output; // if there is an error
    }

    /**
     * opens a file dialog and returns some files
     * Can be used to save or open files
     * @param mode 1 is saving 0 is loading
     * @param workingDir the directory to start at
     * @param name the default name of the file when saving
     * @return array of files selected
     */
    public static File[] fileDialog(int mode, String workingDir, String name, boolean multipleSelect, String[] filters)
    {
        int SAVE = 1;
        int LOAD = 0;
        if (mode != SAVE && mode != LOAD)
            mode = 0;

        Frame f = new Frame();
        FileDialog dialog = new FileDialog(f, "", mode);
        FilenameFilter filter = new FilenameFilter()
        {
            @Override
            public boolean accept(File dir, String name)
            {
                // if there are no filters, any file is good
                if (filters.length == 0)
                    return true;

                // check if it ends with any of the filters
                for(int i = 0; i < filters.length; i++)
                    if(name.toLowerCase().endsWith(filters[i]))
                        return true;
                return false;
            }
        };

        dialog.setFilenameFilter(filter);

        if (workingDir != null)
            dialog.setDirectory(workingDir);
        if (mode == SAVE)
        {
            if (name != null)
                dialog.setFile(name);
        }
        else
        {
            if (multipleSelect)
                dialog.setMultipleMode(true);
            else
                dialog.setMultipleMode(false);
        }

        dialog.setVisible(true);

        return dialog.getFiles();
    }

    /**
     * opens a new finder window at a specific location
     * @param finderOpenPath where to open the finder window at. It can be null
     */
    public static void openFinder(String finderOpenPath)
    {
        // if no path is provided, opens desktop
        if (finderOpenPath == null)
            exec(new String[] {"open", System.getProperty("user.home") + "/Desktop"}, null);
        else
            exec(new String[] {"open", "-R", finderOpenPath}, null);
    }
}



