import sys
import os
import shutil

def process_directory(src_directory, dest_directory):
  contents = [name for name in os.listdir(src_directory) if name[0] != '.']

  if 'annotations.txt' in contents:
    
    other_contents = [name for name in contents if name != 'annotations.txt']

    for name in other_contents:
      
      src_path = src_directory + '/' + name
      dest_path = dest_directory + '/' + name

      shutil.copy(src_path, dest_path)

    src_annotations = src_directory + '/annotations.txt'
    dest_annotations = dest_directory + '/annotations.txt'

    src_f = open(src_annotations, 'r')
    dest_f = open(dest_annotations, 'w')

    for line in src_f:
      
      columns = line.strip().split(' ')

      x_min = int(columns[1])
      x_max = int(columns[3])
      y_min = int(columns[2])
      y_max = int(columns[4])

      x_mid = (x_min + x_max) / 2
      y_mid = (y_min + y_max) / 2

      new_line = ' '.join(columns[:1] + [str(x_mid), str(y_mid)] + columns[5:]) + '\n'
      dest_f.write(new_line)

    src_f.close()
    dest_f.close()

  else:

    for name in contents:

      src_path = src_directory + '/' + name
      dest_path = dest_directory + '/' + name

      os.mkdir(dest_path)

      process_directory(src_path, dest_path)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print 'Need source and destination directories (in that order) as arguments'
  else:

    src_directory = sys.argv[1]
    dest_directory = sys.argv[2]

    os.mkdir(dest_directory)

    process_directory(src_directory, dest_directory)
