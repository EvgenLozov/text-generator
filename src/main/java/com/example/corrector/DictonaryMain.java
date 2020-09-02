package com.example.corrector;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class DictonaryMain {
    public static void main(String[] args) throws IOException {
        File dictionaryFile = new File("corncob_60k_words.txt");

        List<String> allWords = Files.lines(dictionaryFile.toPath()).collect(Collectors.toList());

        Collections.shuffle(allWords);

        AtomicInteger fileNameSuffix = new AtomicInteger(1);
        Lists.partition(allWords, 10_000)
                .forEach(list -> writeToFile("/dictonary/words_en_" + fileNameSuffix.getAndIncrement() + ".txt", list));

    }

    private static void writeToFile(String fileName, List<String> list) {
        try {
            FileUtils.writeLines(new File(fileName), list);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
