for d in */; do
	cd $d
	filename="$(echo $d | cut -d'/' -f1)"".fds"
	~/NIST/fds/Build/impi_intel_linux/fds_impi_intel_linux $filename
	cd ..
done

