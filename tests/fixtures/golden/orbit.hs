!INTERFILE :=
!imaging modality := nucmed
!originating system := simind
!version of keys := 3.3
;program author := M Ljungberg, Lund University

!GENERAL DATA :=
!data offset in bytes := 0
!name of data file := output.a00
;patient name := test_patient

!GENERAL IMAGE DATA :=
!type of data := tomographic
imagedata byte order := LITTLEENDIAN
energy window lower level[1] := 126.16
energy window upper level[1] := 36.84
!number of energy windows := 1
!matrix size [1] := 128
!matrix size [2] := 128
!number format := float
!number of bytes per pixel := 4
scaling factor (mm/pixel) [1] := 4.419600
scaling factor (mm/pixel) [2] := 4.419600

!SPECT STUDY (General) :=
!extent of rotation := 360
!process status := acquired
!number of projections := 60
number of time frames := 1
image duration (sec) [1] := 60.000000

!SPECT STUDY (acquired data) :=
orbit := non-circular
;# Non-Uniform Orbit File := output.cor
Radii := {150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0}
!direction of rotation := CW
start angle := 180.0

!END OF INTERFILE :=
