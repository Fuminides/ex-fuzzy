@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "livehtml" goto livehtml

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  html       to make standalone HTML files
echo.  dirhtml    to make HTML files named index.html in directories  
echo.  singlehtml to make a single large HTML file
echo.  pickle     to make pickle files
echo.  json       to make JSON files
echo.  htmlhelp   to make HTML files and an HTML help project
echo.  qthelp     to make HTML files and a qthelp project
echo.  devhelp    to make HTML files and a Devhelp project
echo.  epub       to make an epub
echo.  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter
echo.  text       to make text files
echo.  man        to make manual pages
echo.  texinfo    to make Texinfo files
echo.  gettext    to make PO message catalogs
echo.  changes    to make an overview over all changed/added/deprecated items
echo.  xml        to make Docutils-native XML files
echo.  pseudoxml  to make pseudoxml-XML files for display purposes
echo.  linkcheck  to check all external links for integrity
echo.  doctest    to run all doctests embedded in the documentation
echo.  coverage   to run coverage check of the documentation
echo.  clean      to clean build directory
echo.  livehtml   to build docs and start live-reload server
goto end

:clean
rmdir /s /q "%BUILDDIR%"
echo.Build directory cleaned.
goto end

:livehtml
sphinx-autobuild "%SOURCEDIR%" "%BUILDDIR%\html" --open-browser --reload-dirs "%SOURCEDIR%"
goto end

:end
popd
