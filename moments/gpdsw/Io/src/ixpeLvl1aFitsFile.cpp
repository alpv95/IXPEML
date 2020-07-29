/***********************************************************************
Copyright (C) 2017--2019 the Imaging X-ray Polarimetry Explorer (IXPE) team.

For the license terms see the file LICENSE, distributed along with this
software.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
***********************************************************************/


#include "__version__.h"
#include "Time/include/ixpeTime.h"
#include "Utils/include/ixpeLogging.h"
#include "Utils/include/ixpeMath.h"
#include "Geometry/include/ixpeHexagonalCoordinates.h"
#include "Recon/include/ixpeTrack.h"
#include "Io/include/ixpeFitsDataFormat.h"
#include "Io/include/ixpeLvl1aFitsFile.h"
#include "Utils/include/ixpeProgressBar.h"


/*!
  Constructor
 */
ixpeLvl1aFitsFile::ixpeLvl1aFitsFile(const std::string& filePath,
                                     const FileMode mode,
                                     bool withOptionalFields) :
  ixpeFitsFile(),
  m_withOptionalFields(withOptionalFields)
{
  m_fileHeader = ixpeFitsDataFormat::ixpeLvl1aFileHeader();
  ixpeFitsFile::switchOpenCreate(filePath, mode);
}


/*!
  Open an existing file. This is a derived version of the base class 'open'
  routine, that allows us to initialize the specific extensions for this file.
 */
void ixpeLvl1aFitsFile::open(const std::string& filePath, bool readAndWrite)
{
  ixpeFitsFile::open(filePath, readAndWrite);
  ixpeFitsFile::addTableExtension(ixpeFitsDataFormat::eventsExtensionName(),
      ixpeFitsDataFormat::ixpeLvl1aEventsExtension(m_withOptionalFields));
  if (checkHdu(ixpeFitsDataFormat::gtiExtensionName())) {
    ixpeFitsFile::addTableExtension(ixpeFitsDataFormat::gtiExtensionName(),
        ixpeFitsDataFormat::ixpeLvl1aGtiExtension());
  }
}

/*!
  Create a new file.
  Note that we *do not* create the GTI binary table here, as we want the
  table to be at the end of the file, but this is interfering destructively
  with the variable-lenght portion of the EVENTS extension, the result being
  that if we create the GTI extension here, and we do it after the EVENTS
  extension, the ixpemdat2fits converter runs ~20 time slower.
  See https://bitbucket.org/ixpesw/gpdsw/issues/165/ixpemdat2fits-slow-on-the-master-branch for more details.
 */
void ixpeLvl1aFitsFile::create(const std::string& filePath, bool overwrite)
{
  ixpeFitsFile::create(filePath, overwrite);
  createPrimaryHdu();
  createEventsBinaryTableExtension();
}

/*!
  Retrieve the file type
 */
FileType ixpeLvl1aFitsFile::fileType()
{
  if (checkHdu(ixpeFitsDataFormat::mcExtensionName())) {
    return FileType::LVL1AMC;
  }
  else {
    return FileType::LVL1A;
  }
}

/*!
  Retrieve the file version
 */
short ixpeLvl1aFitsFile::fileVersion() const
{
  return m_fileHeader.value<short>("LV1_VER");
}

/*!
  Writes the header of the given HDU.
  If the HDU does not already exists in the file, create it.
  Note: The EVENTS HDU is mandatory for this file and should already
  exist, so we let the function fail with an exception if it doesn't.
 */
void ixpeLvl1aFitsFile::writeHeader(const std::string& hduName,
                                    const ixpeFitsHeader& header)
{
  // If the extension does not exists creates it...
  if (!hasTableExtension(hduName)) {
    if (hduName == ixpeFitsDataFormat::mcExtensionName()) {
      createMcBinaryTableExtension();
    } else if (hduName == ixpeFitsDataFormat::gtiExtensionName()) {
      createGtiBinaryTableExtension();
    } else {
      std::stringstream errorMsg;
      errorMsg << "Unkown HDU " << hduName;
      throw(std::runtime_error(errorMsg.str()));
    }
  }
  ixpeFitsFile::writeExtensionHeader(hduName, header);
}

/*!
  Write the header of the primary HDU
 */
void ixpeLvl1aFitsFile::writeFileHeader(const ixpeFitsHeader& lvl1Header)
{
  // Copy all the fields from the Lvl1 input header into m_fileHeader
  try {
    m_fileHeader.update(lvl1Header);
  }
  catch (std::exception& except) {
    LOG_ERROR << except.what();
  }
  // Overwrite the creation date.
  m_fileHeader.setKeyword("DATE", ixpeTime::currentDatetime());
  // Overwrite the file level keyword
  m_fileHeader.setKeyword("LV1_VER", ixpeFitsDataFormat::lv1Version());
  // Now write everything to file
  ixpeFitsFile::writeFileHeader();
}

/*!
  Write the header of the EVENTS binary table extension, merging the
  info from the lvl1 file with the info from ixpeReconConfiguration
 */
void ixpeLvl1aFitsFile::writeEventsHeader(const ixpeFitsHeader& lvl1Header)
{
  const std::string& extName = ixpeFitsDataFormat::eventsExtensionName();
  auto& evtHeader = header(extName);
  // Copy all the fields from the Lvl1 input header
  try {
    evtHeader.update(lvl1Header);
  }
  catch (std::exception& except) {
    LOG_ERROR << except.what();
  }
  typedef ixpeReconConfiguration rc;
  evtHeader.setKeyword("LV1_VER", ixpeFitsDataFormat::lv1Version());
  evtHeader.setKeyword("D_MIN", rc::dmin);
  evtHeader.setKeyword("D_MAX", rc::dmax);
  evtHeader.setKeyword("W_SCALE", rc::weightScale);
  evtHeader.setKeyword("REC_THR", rc::zeroSupThreshold);
  evtHeader.setKeyword("MOM1_THR", rc::moma1Threshold);
  evtHeader.setKeyword("MOM2_THR", rc::moma2Threshold);
  evtHeader.setKeyword("REC_VER", std::string(__GPDSW_VERSION__));
  evtHeader.setKeyword("ACOLCORR", rc::coherentNoiseCorrectionOffset);
  evtHeader.setKeyword("ATRGCORR", rc::triggerMiniclusterCorrectionOffset);
  if (rc::writeTracks) {
    evtHeader.setKeyword("TRK_FULL", TRUE);
  }
  else {
    evtHeader.setKeyword("TRK_FULL", FALSE);
  }
  ixpeFitsFile::writeHeader(extName);
}

/*!
  Write the header of the GTI binary table extension, copying the
  info from the lvl1 file
 */
void ixpeLvl1aFitsFile::writeGtiHeader(const ixpeFitsHeader& lvl1Header)
{
  const std::string& extName = ixpeFitsDataFormat::gtiExtensionName();
  // Create the corresponding extension if it does not exists
  if (!checkHdu(extName)) {
    createGtiBinaryTableExtension();
  }
  auto& gtiHeader = header(extName);
  // Copy all the fields from the Lvl1 input header
  try {
    gtiHeader.update(lvl1Header);
  }
  catch (std::exception& except) {
    LOG_ERROR << except.what();
  }
  gtiHeader.setKeyword("LV1_VER", ixpeFitsDataFormat::lv1Version());
  // Write it to file
  ixpeFitsFile::writeHeader(extName);
}

/*!
  Create the 'EVENTS' binary table object
 */
void ixpeLvl1aFitsFile::createEventsBinaryTableExtension()
{
  const auto& extName = ixpeFitsDataFormat::eventsExtensionName();
  createBinaryTableExtension(extName,
      ixpeFitsDataFormat::ixpeLvl1aEventsExtension(m_withOptionalFields));
  header(extName).setKeyword("LV1_VER", ixpeFitsDataFormat::lv1Version());
}

/*!
  Create the 'GTI' binary table object
 */
void ixpeLvl1aFitsFile::createGtiBinaryTableExtension()
{
  const auto& extName = ixpeFitsDataFormat::gtiExtensionName();
  createBinaryTableExtension(extName,
                             ixpeFitsDataFormat::ixpeLvl1aGtiExtension());
  header(extName).setKeyword("LV1_VER", ixpeFitsDataFormat::lv1Version());
}

/*!
  Create the 'MONTE_CARLO' binary table object
 */
void ixpeLvl1aFitsFile::createMcBinaryTableExtension()
{
  createBinaryTableExtension(ixpeFitsDataFormat::mcExtensionName(),
                             ixpeFitsDataFormat::ixpeLvl1aMcExtension());
}

/*!
  Write an event in the fits file.
  The event is written at the row number specified by the m_currentRow
  member variable, which is then incremented by one.
  Now need to write whole fits file at once, but then modulaton_factor.py wont be able to read properly??
 */
void ixpeLvl1aFitsFile::write(int eventId, const ixpeEvent& event,
                              const std::vector<ixpeTrack>& tracks)
{
  const std::string& extName = ixpeFitsDataFormat::eventsExtensionName();
  // Move to the EVENTS HDU
  selectHdu(extName);
  
  //auto bar = ixpeProgressBar(500);

  //for (int eventId : eventIds) {
  //ixpeEvent event = events[eventId];
  //std::vector<ixpeTrack> tracks = vectorTracks[eventId];

  // Increment the current row pointer to write on the next row
  //tableExtension(extName).setCurrentRow(eventId+1);

  tableExtension(extName).incrementCurrentRow();
  writeCell("PAKTNUMB", event.packetId());
  writeCell("TRG_ID", event.triggerId());
  writeCell("TIME", event.timestamp());
  writeCell("LIVETIME", event.livetime());
  writeCell("ROI_SIZE", event.size());
  writeCell("MIN_CHIPX", event.minColumn());
  writeCell("MAX_CHIPX", event.maxColumn());
  writeCell("MIN_CHIPY", event.minRow());
  writeCell("MAX_CHIPY", event.maxRow());
  writeCell("ERR_SUM", event.errorSummary());
  writeCell("DU_STATUS", event.duStatus());
  writeCell("DSU_STATUS", event.dsuStatus());
  writeCell("STATUS", event.status());
  writeCell("NUM_PIX", event.numAboveThresholdPixels());
  //if (m_withOptionalFields) {
  //  writeCell("PIX_PIS",
  //           const_cast<std::vector<event_adc_counts_t>&>(event.adcCounts())); //[track.hits() for track in tracks]
  //  writeCell("PIX_TRK",
  //           const_cast<std::vector<cluster_id_t>&>(event.clusterIds()));
  //}
  int numTracks = static_cast<int>(tracks.size());
  writeCell("NUM_CLU", numTracks);
  if (numTracks ==  0){
    writeEmptyTrack();
  }
  else{
    const ixpeTrack& mainTrack = tracks.at(0);
    // We need to write EVT_FRA here, because it requires event info
    if (event.totalPulseInvariant() > 0) {
      writeCell<float>("EVT_FRA", mainTrack.pulseInvariant()/
                        event.totalPulseInvariant());
    } else {
      writeCell<float>("EVT_FRA", 1.);
    }
    writeMainTrack(mainTrack);

    if (m_withOptionalFields) {
       std::vector<float> pix_x(400, std::nan(""));
       std::vector<float> pix_y(400, std::nan(""));
       std::vector<event_adc_counts_t> pix_adc(400, -1);
       int hitcount = 0;
       for (auto& hit : mainTrack.hits()) {
          if (hitcount < 400) {
          pix_x[hitcount] = hit.x;
          pix_y[hitcount] = hit.y;
          pix_adc[hitcount] = hit.pulseHeight;
          hitcount++;
          } else {
            break;
          }
       }
       writeCell("PIX_X", const_cast<std::vector<float>&>(pix_x));
       writeCell("PIX_Y", const_cast<std::vector<float>&>(pix_y));
       writeCell("PIX_PHA", const_cast<std::vector<event_adc_counts_t>&>(pix_adc));
    }
  }
  //bar.advance();
  //}
  //bar.setDone();
}

/*!
  Read a GTI table
 */
ixpeGtiTable ixpeLvl1aFitsFile::gtiTable()
{
  const std::string& extName = ixpeFitsDataFormat::gtiExtensionName();
  // Create the corresponding extension if it does not exists
  if (!checkHdu(extName)) {
    createGtiBinaryTableExtension();
  }
  // Intialize the empty table
  ixpeGtiTable gtiTable;
  // Move to the GTI HDU
  selectHdu(extName);
  // Get the number of rows
  int rows = numRows();
  // Start reading from the first row
  int currentRow = 1;
  tableExtension(extName).setCurrentRow(currentRow);
  double start = -1.;
  double stop = -1.;
  // Loop over the rows and fill the table
  while (currentRow <= rows) {
    start = readCell<double>("START");
    stop = readCell<double>("STOP");
    gtiTable.push_back(std::make_pair(start, stop));
    // Increment the current row pointer
    currentRow = tableExtension(extName).incrementCurrentRow();
  }
  return gtiTable;
}

/*!
  Write a GTI table
 */
void ixpeLvl1aFitsFile::writeGtiTable(const ixpeGtiTable& gtiTable)
{
  const std::string& extName = ixpeFitsDataFormat::gtiExtensionName();
  // Create the corresponding extension if it does not exist
  if (!checkHdu(extName)) {
    createGtiBinaryTableExtension();
  }
  // Move to the GTI HDU
  selectHdu(extName);
  // Start writing from the first row
  tableExtension(extName).setCurrentRow(1);
  for (const auto& gti : gtiTable) {
    writeCell("START", gti.first);
    writeCell("STOP", gti.second);
    tableExtension(extName).incrementCurrentRow();
  }
}

/*!
  Write main track info.
 */
void ixpeLvl1aFitsFile::writeMainTrack(const ixpeTrack& mainTrack)
{
  writeCell("TRK_SIZE", mainTrack.numHits());
  writeCell("TRK_BORD", mainTrack.numEdgePixels());
  writeCell("PHA", mainTrack.pulseHeight());
  // Since, starting from v4 of the LV1 data format, we reduced the precision
  // of the following quantites from TDOUBLE to TFLOAT, here we need to
  // explicitly invoke the float version of the writing routine, in order to
  // force a downcast of the related quantities (which are represented
  // with a double precision in the code)
  writeCell<float>("SN", mainTrack.signalToNoiseRatio());
  writeCell<float>("PI", mainTrack.pulseInvariant());
  writeCell<float>("PHI1", mainTrack.firstPassMomentsAnalysis().phi());
  writeCell<float>("PHI2", mainTrack.secondPassMomentsAnalysis().phi());
  writeCell<float>("DETPHI", mainTrack.secondPassMomentsAnalysis().phi());
  writeCell<float>("DETX", mainTrack.absorptionPoint().x());
  writeCell<float>("DETY", mainTrack.absorptionPoint().y());
  writeCell<float>("BARX", mainTrack.barycenter().x());
  writeCell<float>("BARY", mainTrack.barycenter().y());
  writeCell<float>("TRK_M2T", mainTrack.firstPassMomentsAnalysis().mom2trans());
  writeCell<float>("TRK_M2L", mainTrack.firstPassMomentsAnalysis().mom2long());
  writeCell<float>("TRK_M3L", mainTrack.firstPassMomentsAnalysis().mom3long());
  writeCell<float>("TRK_SKEW", mainTrack.firstPassMomentsAnalysis().skewness());
}

/*!
  Write default values for events with no valid tracks.
 */
void ixpeLvl1aFitsFile::writeEmptyTrack()
{
  writeCell("TRK_SIZE", 0);
  writeCell("TRK_BORD", 0);
  writeCell("PHA", 0);
  // Since, starting from v4 of the LV1 data format, we reduced the precision
  // of the following quantites from TDOUBLE to TFLOAT, here we need to
  // explicitly invoke the float version of the writing routine, in order to
  // force a downcast of the related quantities (which are represented
  // with a double precision in the code)
  writeCell<float>("PI", 0.);
  writeCell<float>("EVT_FRA", 0.);
  writeCell<float>("SN", 0.);
  writeCell<float>("PHI1", FLOATNULLVALUE);
  writeCell<float>("PHI2", FLOATNULLVALUE);
  writeCell<float>("DETPHI", FLOATNULLVALUE);
  writeCell<float>("DETX", FLOATNULLVALUE);
  writeCell<float>("DETY", FLOATNULLVALUE);
  writeCell<float>("BARX", FLOATNULLVALUE);
  writeCell<float>("BARY", FLOATNULLVALUE);
  writeCell<float>("TRK_M2T", FLOATNULLVALUE);
  writeCell<float>("TRK_M2L", FLOATNULLVALUE);
  writeCell<float>("TRK_M3L", FLOATNULLVALUE);
  writeCell<float>("TRK_SKEW", FLOATNULLVALUE);
}
