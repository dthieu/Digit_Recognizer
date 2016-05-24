
/*Usage: DigitRecognizer.exe Models Rate Function FileNameAction FileNameAfter [FileResult]
+ Models [0/1]: bayes/SVM 
+ Rate [0/1/2]: LOW/ MEDIUM/ HIGHT
	- HOG extract feature with:
		...RateOfCellSize: ratio between WindowSize and CellSize
		...CellSize = WindowSize / N.
		...RateOfCellSize: LOW -> N = 2, MEDIUM -> N = 4, HIGH-> N = 8
		...blockSize = CellSize * 2
		...BlockStride = CellSize


+ Function: [0/1] train/Test

+ if function is 0 (Train) then:
	fileNameAction: fileName of data Train (*.csv)
	fileNameAfter: fileName save dataTrain (*.xml)

+ else function is 1 (Test):

	filenameAction: fileName of data Test (*.csv)
	fileNameAfter: fileName save dataTrain (*.xml)
	fileResult: fileName result of program. (*.csv)
*/

note: Với Bayes thì nên đặt ở rate là LOW hoặc MEDIUM. Bởi số chiều đặc trưng ảnh hưởng đến quá trình train và có thể gây lỗi.