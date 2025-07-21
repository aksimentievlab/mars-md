///////////////////////////////////////////////////////////////////////  
// Author: Jeff Comer <jcomer2@illinois.edu>
#ifndef TRAJECTORYWRITER_H
#define TRAJECTORYWRITER_H

#define PDB_TEMPLATE_LINE "ATOM      1  CA  MOL S   1      -6.210  -9.711   3.288  0.00  0.00      ION"

#include <cstdio>
#include "Math/Types.h"
#include "DcdWriter.h"
#include "FileHandle.h"
#include <string>
#include <sstream>
#include <string_view>
#include "ARBDLogger.h"
#include "ARBDException.h"

namespace ARBD{
class TrajectoryWriter {
public:
  const static int formatDcd = 0;
  const static int formatPdb = 1;
  const static int formatTraj = 2;

  TrajectoryWriter(const char* filePrefix, const char* formatName, Matrix3 box0, int num0, float timestep0, int outputPeriod0) 
    : box(box0), num(num0), timestep(timestep0), outputPeriod(outputPeriod0)  {
    pdbTemplate = PDB_TEMPLATE_LINE;
    format = getFormatCode(std::string(formatName));
    makeUnitCell();

    fileName = filePrefix;
    fileName += ".";
    fileName += getFormatName(format);
    
    if (format == formatDcd) {
      dcd = new DcdWriter(fileName);
      dcd->writeHeader(fileName.c_str(), num, 1, 0, outputPeriod, 0, timestep, 1);
    }
  }

  ~TrajectoryWriter() {
    if (format == formatDcd) delete dcd;
  }

private:
  Matrix3 box;
  std::string fileName;
  FileHandle input_file(fileName, "r");
  FILE* fp = filename.get();
  int format;
  std::string pdbTemplate;
  double unitCell[6];		/* use double for dcd format */
  int num;
  float timestep;
  int outputPeriod;
  DcdWriter* dcd;

  void makeUnitCell() {
    float pi = 4.0f*atan(1.0f);

    unitCell[0] = box.ex().length();
    unitCell[2] = box.ey().length();
    unitCell[5] = box.ez().length();

    double bc = box.ey().dot(box.ez());
    double ac = box.ex().dot(box.ez());
    double ab = box.ex().dot(box.ey());

    unitCell[1] = bc/unitCell[0]/unitCell[2]/pi*180.0f;
    unitCell[3] = ac/unitCell[0]/unitCell[5]/pi*180.0f;
    unitCell[4] = ab/unitCell[2]/unitCell[5]/pi*180.0f;

    unitCell[1] = unitCell[1] > 1.0 ? 1.0 : unitCell[1] < -1.0 ? -1.0 : unitCell[1];
    unitCell[3] = unitCell[3] > 1.0 ? 1.0 : unitCell[3] < -1.0 ? -1.0 : unitCell[3];
    unitCell[4] = unitCell[4] > 1.0 ? 1.0 : unitCell[4] < -1.0 ? -1.0 : unitCell[4];
  }

public:
  static int getFormatCode(std::string format) {
    format.lower();
    if (format == std::string("dcd")) return formatDcd;
    if (format == std::string("pdb")) return formatPdb;
    if (format == std::string("traj")) return formatTraj;
    return formatDcd;
  }

  static std::string getFormatName(int formatCode) {
    switch(formatCode) {
    case formatPdb:
      return std::string("pdb");
    case formatTraj:
      return std::string("traj");
    case formatDcd:
      return std::string("dcd");
    default:
      return std::string("dcd");
    }
  }

  void newFile(const Vector3* pos, const std::string* name, float t, int n) const {
    switch(format) {
    case formatPdb:
      newPdb(fileName, pos, name);
      break;
    case formatTraj:
      newTraj(pos, name, t, n);
      break;
    case formatDcd:
    default:
      newDcd(pos, name);
      break;
    }
  }

  void newFile(const Vector3* pos, const std::string* name, const int* id, float t, int n) const {
    switch(format) {
    case formatPdb:
      newPdb(fileName, pos, name);
      break;
    case formatTraj:
      newTraj(pos, name, id, t, n);
      break;
    case formatDcd:
    default:
      newDcd(pos, name);
      break;
    }
  }

  void append(const Vector3* pos, const std::string* name, float t, int n) const {
    switch(format) {
    case formatPdb:
      appendPdb(pos, name);
      break;
    case formatTraj:
      appendTraj(pos, name, t, n);
      break;
    case formatDcd:
    default:
      appendDcd(pos);
      break;
    }
  }
  
  void append(const Vector3* pos, const std::string* name, const int* id, float t, int n) const {
    switch(format) {
    case formatPdb:
      appendPdb(pos, name);
      break;
    case formatTraj:
      appendTraj(pos, name, id, t, n);
      break;
    case formatDcd:
    default:
      appendDcd(pos);
      break;
    }
  }

  void newPdb(const char* outFile, const Vector3* pos, const std::string* name) const {
    char s[128];

    LOGINFO(s, "CRYST1   %.3f   %.3f   %.3f  90.00  90.00  90.00 P 1           1\n", box.xx, box.yy, box.zz);
    std::string sysLine(s);

    LOGINFO(s, "REMARK   frameTime %.10g ns\n", outputPeriod*timestep);
    std::string remarkLine(s);
    
    std::string line;

    FILE* out = std::fopen(outFile, "w");
    LOGINFO(out, "%s", sysLine.val());
    LOGINFO(out, "%s", remarkLine.val());

    for (int i = 0; i < num; i++) {
      line = makePdbLine(pdbTemplate, i, name[i], i, name[i], pos[i], 0.0);
      LOGINFO(out, "%s",  line.val());
      LOGINFO(out, "\n");
    }
    LOGINFO(out, "END\n");
    fclose(out);
  }

  void appendPdb(const Vector3* pos, const std::string* name) const {
    std::string line;

    FILE* out = std::fopen(fileName, "a");
    for (int i = 0; i < num; i++) {
      line = makePdbLine(pdbTemplate, i, name[i], i, name[i], pos[i], 0.0);
      LOGINFO(out, "%s", line.val());
      LOGINFO(out, "\n");
    }
    LOGINFO(out, "END\n");
    fclose(out);
  }
 
  void newTraj(const Vector3* pos, const std::string* name, float t, int n) const {
    FILE* out = std::fopen(fileName, "w");
    for (int i = 0; i < n; i++)
      LOGINFO(out, "%s %.10g %.10g %.10g %.10g\n", name[i].val(), t, pos[i].x, pos[i].y, pos[i].z);
    LOGINFO(out, "END\n");
    fclose(out);
  }

  void newTraj(const Vector3* pos, const std::string* name, const int* id, float t, int n) const {
    FILE* out = std::fopen(fileName, "w");
    for (int i = 0; i < n; i++)
      LOGINFO(out, "%s %.10g %.10g %.10g %.10g %d\n", name[i].val(), t, pos[i].x, pos[i].y, pos[i].z, id[i]);
    LOGINFO(out, "END\n");
    fclose(out);
  }

  void appendTraj(const Vector3* pos, const std::string* name, float t, int n) const {
    FILE* out = std::fopen(fileName, "a");
    for (int i = 0; i < n; i++)
      LOGINFO(out, "%s %.10g %.10g %.10g %.10g\n", name[i].val(), t, pos[i].x, pos[i].y, pos[i].z);
    LOGINFO(out, "END\n");
    fclose(out);
  }

  void appendTraj(const Vector3* pos, const std::string* name, const int* id, float t, int n) const {
    FILE* out = std::fopen(fileName, "a");
    for (int i = 0; i < n; i++)
      LOGINFO(out, "%s %.10g %.10g %.10g %.10g %d\n", name[i].val(), t, pos[i].x, pos[i].y, pos[i].z, id[i]);
    LOGINFO(out, "END\n");
    fclose(out);
  }

  void newDcd(const Vector3* pos, const std::string* name) const {
    /*  
    // Write a new pdb to store the atom names and such.
    char pdbFile[128];
    sprintf(pdbFile, "%s.pdb", fileName.val());
    newPdb(pdbFile, pos, name);
    */

    // Write first frame.
    appendDcd(pos);
  }

  void appendDcd(const Vector3* pos) const {
    float* x = new float[num];
    float* y = new float[num];
    float* z = new float[num];

    for (int i = 0; i < num; i++) {
      x[i] = pos[i].x;
      y[i] = pos[i].y;
      z[i] = pos[i].z;
    }
    dcd->writeStep(num, x, y, z, unitCell);

    delete[] x;
    delete[] y;
    delete[] z;
  }

  static std::string makePdbLine(const std::string& tempLine, int index, const std::string& segName, int resId, 
			    const std::string& name, Vector3 r, float beta) {
    char s[128];

    std::string record("ATOM  ");
    sprintf(s, "     %5i ", index);
    std::string si = std::string(s).range(-6,-1);
    if (name.length() == 4) sprintf(s, "%s   ", name.val());
    else sprintf(s, " %s   ", name.val());
    std::string nam = std::string(s).range(0,3);
    std::string temp0 = tempLine.range(16,21);
  
    sprintf(s, "    %d", resId);
    std::string res = std::string(s).range(-4,-1);
    std::string temp1 = tempLine.range(26,29);
  
    sprintf(s,"       %.3f", r.x);
    std::string sx = std::string(s).range(-8,-1);
    sprintf(s,"       %.3f", r.y);
    std::string sy = std::string(s).range(-8,-1);
    sprintf(s,"       %.3f", r.z);
    std::string sz = std::string(s).range(-8,-1);

    std::string temp2 = tempLine.range(54,59);
    sprintf(s,"    %.2f", beta);
    std::string bet = std::string(s).range(-6,-1);
    std::string temp3 = tempLine.range(66,71);

    sprintf(s, "%s    ", segName.val());
    std::string seg = std::string(s).range(0,3);

    std::string ret(record);
    ret.add(si);
    ret.add(nam);
    ret.add(temp0);
    ret.add(res);
    ret.add(temp1);
    ret.add(sx);
    ret.add(sy);
    ret.add(sz);
    ret.add(temp2);
    ret.add(bet);
    ret.add(temp3);
    ret.add(seg);
  
    return ret;
  }
};
}
#endif
