package main.java;

import main.java.Utilities.Matrix;

import java.io.FileNotFoundException;
import java.util.ArrayList;

/**
 * the runner for preprocess
 *
 * @author Amos Decker
 * @version 1.0
 */
public class preProcessRunner
{
    public static void main(String[] args) throws FileNotFoundException, java.io.IOException
    {
        runPreprocess("senate");
    }

    public static void runPreprocess(String topic) throws FileNotFoundException, java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir")  + "/src";
        Preprocess p = new Preprocess(pathToProject + "/main/java/" + topic + ".txt");
        StringBuilder contents = p.readContents();

        ArrayList<Matrix> matrices = p.cleanEncode(contents);

        System.out.println("done with preprocess starting saving");

        for (int m = 0; m < matrices.size(); m++)
        {
            matrices.get(m).save(pathToProject + "/main/java/" + topic + "Saved.txt");
        }
        matrices.clear(); // empty out the array b/c it is full of stuff that I don't need
        System.out.println("done with saving");
    }
}
