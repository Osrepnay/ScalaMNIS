name := "ScalaMNIST"

version := "0.1"

scalaVersion := "2.13.4"

libraryDependencies ++= Seq("com.github.haifengl" %% "smile-scala" % "2.6.0",
	"org.bytedeco" % "javacpp" % "1.5.4" classifier "macosx-x86_64" classifier "windows-x86_64" classifier
		"linux-x86_64" classifier "linux-arm64" classifier "linux-ppc64le" classifier "android-arm64" classifier
		"ios-arm64",
	"org.bytedeco" % "openblas" % "0.3.10-1.5.4" classifier "macosx-x86_64" classifier "windows-x86_64" classifier
		"linux-x86_64" classifier "linux-arm64" classifier "linux-ppc64le" classifier "android-arm64" classifier
		"ios-arm64",
	"org.bytedeco" % "arpack-ng" % "3.7.0-1.5.4"  classifier "macosx-x86_64" classifier "windows-x86_64" classifier
		"linux-x86_64" classifier "linux-arm64" classifier "linux-ppc64le")