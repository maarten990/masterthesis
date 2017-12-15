package cmdline

import javafx.application.Application

fun main(args: Array<String>) {
    if (args.getOrElse(0, {""}) == "gui") {
        Application.launch(gui.ClusterApp::class.java, *args)
    } else {
        println("Hello World!")
    }
}

