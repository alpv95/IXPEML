/***********************************************************************
Copyright (C) 2017 the Imaging X-ray Polarimetry Explorer (IXPE) team.

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

/*!
  @file
  @brief Write/read reconstruction info into/from fits files
*/


#ifndef IXPELVL1AFITSFILE_H
#define IXPELVL1AFITSFILE_H


#include "Io/include/ixpeInputFile.h"
#include "Io/include/ixpeFitsFile.h"
#include "Event/include/ixpeEvent.h"
#include "MonteCarlo/include/ixpeMcInfo.h"
#include "Recon/include/ixpeReconConfiguration.h"
#include "Recon/include/ixpeEvtStatusBitMask.h"



// Forward declarations:
class ixpeTrack;


//! Class handling reading/writing operations to ixperecon output fits files.
class ixpeLvl1aFitsFile : public ixpeFitsFile
{

 public:

  //! Constructor.
  ixpeLvl1aFitsFile(const std::string& filePath,
                    const FileMode mode=FileMode::READ,
                    bool withOptionalFields = false);

  //! Open an existing file
  virtual void open(const std::string& filePath, bool readAndWrite);

  //! Create a new file
  virtual void create(const std::string& filePath, bool overwrite = false);

  //! Retrieve the file type
  virtual FileType fileType();

  //! Writes to disk the given header in the given HDU
  void writeHeader(const std::string& hduName,
                   const ixpeFitsHeader& header);

  //! Read the gti table from this file
  ixpeGtiTable gtiTable();

  //! Write the header primary HDU
  void writeFileHeader(const ixpeFitsHeader& lvl1Header);

  //! Write the header of the EVENTS binary table extension
  void writeEventsHeader(const ixpeFitsHeader& lvl1Header);

  //! Write the header of the GTI binary table extension
  void writeGtiHeader(const ixpeFitsHeader& lvl1Header);

  //! Write an event into the main binary table
  //void write(std::vector<int>& eventIds, const std::vector<ixpeEvent>& events,
  //           const std::vector<std::vector<ixpeTrack>>& vectorTracks);
  void write(int eventId, const ixpeEvent& event,
             const std::vector<ixpeTrack>& tracks,
             ProcStatusMask_t procStatusMask = 0);

  //! Write Monte Carlo info in the current row of the MONTECARLO extension
  void writeMcInfo(const ixpeMcInfo& mcInfo);

  //! Write a GTI table into the file
  void writeGtiTable(const ixpeGtiTable& gtiTable);

  //! Version of the file (basically the content of the LV1_VER header keyword)
  short fileVersion() const;

 private:

  // Version of the file (we use the same number for LV1 and LV1a);
  short m_lv1Version;

  //! Header of the EVENTS binary table extension
  ixpeFitsHeader m_eventsExtensionHeader;

  //! Flag telling if the optional fields have to be written/read
  bool m_withOptionalFields;

  //! Write main track info
  void writeMainTrack(const ixpeTrack& mainTrack);

  //! Write default values for events with no valid tracks.
  void writeEmptyTrack();

  //! Write default values for tracks with failed moments analysis
  void writeFailedMomentsAnalysis();

  //! Creates the EVENTS binary table extension
  void createEventsBinaryTableExtension();

  //! Create the MONTE_CARLO binary table extension
  void createMcBinaryTableExtension();

  //! Creates the GTI binary table extension
  void createGtiBinaryTableExtension();

};


#endif //IXPELVL1AFITSFILE_H
