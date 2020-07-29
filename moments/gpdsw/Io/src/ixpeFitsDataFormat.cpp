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


#include <string>
#include <cstdint>

#include "Io/include/ixpeFitsHeaderSvc.h"
#include "Io/include/ixpeFitsDataFormat.h"


/*!
  Declaration of the keywords of the Lvl1 file header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1FileHeader()
{
  ixpeFitsHeader header;
  ixpeFitsHeaderSvc::addInstrumentKeywords(header);
  ixpeFitsHeaderSvc::addObservationKeywords(header);
  ixpeFitsHeaderSvc::addContactNumberKeyword(header);
  ixpeFitsHeaderSvc::addSourceConfigurationKeyword(header);
  ixpeFitsHeaderSvc::addPacketTypeKeywords(header);
  ixpeFitsHeaderSvc::addApidKeyword(header);
  ixpeFitsHeaderSvc::addTimeKeywords(header);
  ixpeFitsHeaderSvc::addFileDateKeyword(header);
  ixpeFitsHeaderSvc::addObsDateKeywords(header);
  ixpeFitsHeaderSvc::addLv1VersionKeyword(header);
  ixpeFitsHeaderSvc::addOriginKeyword(header);
  ixpeFitsHeaderSvc::addCreatorKeywords(header);
  ixpeFitsHeaderSvc::addFilterWheelKeywords(header);
  ixpeFitsHeaderSvc::addClockCorrectionKeyword(header);
  header.addKeyword<std::string>("FILE_LVL", "LV1", "File level");
  return header;
}

/*!
  Declaration of the keywords of the Lvl1 EVENTS extension header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1EventsExtensionHeader()
{
  ixpeFitsHeader header;
  ixpeFitsHeaderSvc::addInstrumentKeywords(header);
  ixpeFitsHeaderSvc::addObservationKeywords(header);
  ixpeFitsHeaderSvc::addTimeKeywords(header);
  ixpeFitsHeaderSvc::addLv1VersionKeyword(header);
  ixpeFitsHeaderSvc::addDaqKeywords(header);
  ixpeFitsHeaderSvc::addDaqConfigKeywords(header);
  return header;
}

/*!
  Declaration of the keywords of the Lvl1 GTI binary table
  extension header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1GtiExtensionHeader()
{
  ixpeFitsHeader header;
  ixpeFitsHeaderSvc::addInstrumentKeywords(header);
  ixpeFitsHeaderSvc::addObservationKeywords(header);
  ixpeFitsHeaderSvc::addTimeKeywords(header);
  ixpeFitsHeaderSvc::addLv1VersionKeyword(header);
  ixpeFitsHeaderSvc::addDaqKeywords(header);
  return header;
}

/*!
  Declaration of the keywords of the Lv1 MONTE_CARLO binary table
  extension header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1McExtensionHeader()
{
  ixpeFitsHeader header;
  ixpeFitsHeaderSvc::addMonteCarloKeywords(header);
  return header;
}

/*!
 */
void ixpeFitsDataFormat::addLvl1EventsFixedSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  ext.insertField("PAKTNUMB", "1J", " ", TINT);
  ext.insertField("TRG_ID", "1J", " ", TINT);
  ext.insertField("SEC", "1K", " ", TLONGLONG);
  ext.insertField("MICROSEC", "1J", " ", TINT);
  ext.insertField("TIME", "1D", " ", TDOUBLE);
  ext.insertField("LIVETIME", "1J", "", TINT);
  ext.insertField("MIN_CHIPX", "1I", " ", TSHORT);
  ext.insertField("MAX_CHIPX", "1I", " ", TSHORT);
  ext.insertField("MIN_CHIPY", "1I", " ", TSHORT);
  ext.insertField("MAX_CHIPY", "1I", " ", TSHORT);
  ext.insertField("ROI_SIZE", "1J", " ", TINT);
  ext.insertField("ERR_SUM", "1U", " ", TUSHORT);
  ext.insertField("DU_STATUS", "1U", " ", TUSHORT);
  ext.insertField("DSU_STATUS", "1U", " ", TUSHORT);
}

/*!
 */
void ixpeFitsDataFormat::addLvl1EventsVariableSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  ext.insertField("PIX_PHAS", "QI", " ", TSHORT);
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvl1 "EVENTS" binary
  table extension.
*/
ixpeFitsBinaryTableExtension
ixpeFitsDataFormat::ixpeLvl1EventsExtension()
{
  ixpeFitsBinaryTableExtension ext(\
      ixpeFitsDataFormat::ixpeLvl1EventsExtensionHeader());
  ixpeFitsDataFormat::addLvl1EventsFixedSizeFields(ext);
  ixpeFitsDataFormat::addLvl1EventsVariableSizeFields(ext);
  return ext;
}

/*!
 */
void ixpeFitsDataFormat::addLvl1GtiFields(ixpeFitsBinaryTableExtension& ext)
{
  ext.insertField("START", "1D", "seconds", TDOUBLE);
  ext.insertField("STOP", "1D", "seconds", TDOUBLE);
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvl1 "GTI" binary table
  extension.
*/
ixpeFitsBinaryTableExtension ixpeFitsDataFormat::ixpeLvl1GtiExtension()
{
  ixpeFitsBinaryTableExtension ext(
      ixpeFitsDataFormat::ixpeLvl1GtiExtensionHeader());
  ixpeFitsDataFormat::addLvl1GtiFields(ext);
  return ext;
}

/*!
 */
void ixpeFitsDataFormat::addLvl1McFixedSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  ext.insertField("ENERGY", "1D", "keV", TDOUBLE);
  ext.insertField("ABS_X", "1D", "mm", TDOUBLE);
  ext.insertField("ABS_Y", "1D", "mm", TDOUBLE);
  ext.insertField("ABS_Z", "1D", "mm", TDOUBLE);
  ext.insertField("ABS_ELE", "1A", "", TSTRING);
  ext.insertField("PE_ENE", "1D", "keV", TDOUBLE);
  ext.insertField("PE_PHI", "1D", "rad", TDOUBLE);
  ext.insertField("PE_THET", "1D", "rad", TDOUBLE);
  ext.insertField("AUG_ENE", "1D", "keV", TDOUBLE);
  ext.insertField("AUG_PHI", "1D", "rad", TDOUBLE);
  ext.insertField("AUG_THET", "1D", "rad", TDOUBLE);
  ext.insertField("TRK_LEN", "1D", "mm", TDOUBLE);
  ext.insertField("RANGE", "1D", "mm", TDOUBLE);
  ext.insertField("NUM_PAIR", "1J", "", TINT);
}

/*!
 */
void ixpeFitsDataFormat::addLvl1McVariableSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  ext.insertField("ION_POSX", "PD", "mm", TDOUBLE);
  ext.insertField("ION_POSY", "PD", "mm", TDOUBLE);
  ext.insertField("ION_POSZ", "PD", "mm", TDOUBLE);
  ext.insertField("ION_PE", "PL", "", TLOGICAL);
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvl1 "MONTE_CARLO"
  binary table extension.
*/
ixpeFitsBinaryTableExtension ixpeFitsDataFormat::ixpeLvl1McExtension()
{
  ixpeFitsBinaryTableExtension ext (
      ixpeFitsDataFormat::ixpeLvl1McExtensionHeader());
  ixpeFitsDataFormat::addLvl1McFixedSizeFields(ext);
  ixpeFitsDataFormat::addLvl1McVariableSizeFields(ext);
  return ext;
}


/*!
  Declaration of the keywords of the Lv1a file header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1aFileHeader()
{
  // The only difference with the LV1 file header is the value of the FILE_LVL
  // keyword
  ixpeFitsHeader header = ixpeFitsDataFormat::ixpeLvl1FileHeader();
  header.setKeyword("FILE_LVL", "LV1a");
  return header;
}

/*!
  Declaration of the keywords of the Lvl1a EVENTS binary table extension
  header.
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1aEventsExtensionHeader()
{
  // The header of the LVL1a EVENTS extension has all the keyowrds of the Lv1
  // plus a few more containing information about the reconstruction and the
  // calibrations
  ixpeFitsHeader header = ixpeFitsDataFormat::ixpeLvl1EventsExtensionHeader();
  ixpeFitsHeaderSvc::addCaldbVersionKeyword(header);
  ixpeFitsHeaderSvc::addTrkFullKeyword(header);
  ixpeFitsHeaderSvc::addReconVersioKeyword(header);
  ixpeFitsHeaderSvc::addReconOptionsKeywords(header);
  return header;
};

/*!
  Declaration of the keywords of the Lvl1a GTI binary table
  extension header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1aGtiExtensionHeader()
{
  // This is identical to the LV1 GTI extension header
  return ixpeFitsDataFormat::ixpeLvl1GtiExtensionHeader();
}

/*!
  Declaration of the keywords of the Lv1a MONTE_CARLO binary table
  extension header
*/
ixpeFitsHeader ixpeFitsDataFormat::ixpeLvl1aMcExtensionHeader()
{
  // This is identical to the LV1 MONTE_CARLO extension header
  return ixpeFitsDataFormat::ixpeLvl1McExtensionHeader();
}


/*!
 */
void ixpeFitsDataFormat::addLvl1aEventsFixedSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  ixpeFitsDataFormat::addLvl1EventsFixedSizeFields(ext);
  ext.insertField("STATUS", "16X", "", TBIT);
  ext.insertField("NUM_CLU", "1I", "", TSHORT);
  ext.insertField("NUM_PIX", "1I", "", TSHORT);
  ext.insertField("EVT_FRA", "1E", "", TFLOAT);
  ext.insertField("SN", "1E", "", TFLOAT);
  ext.insertField("TRK_SIZE", "1I", "pixels", TSHORT);
  ext.insertField("TRK_BORD", "1I", "pixels", TSHORT);
  ext.insertField("PHA", "1J", "ADC counts", TINT);
  ext.insertField("PI", "1E", "", TFLOAT);
  ext.insertField("PHI1", "1E", "rad", TFLOAT);
  ext.insertField("PHI2", "1E", "rad", TFLOAT);
  ext.insertField("DETPHI", "1E", "rad", TFLOAT);
  ext.insertField("DETX", "1E", "mm", TFLOAT);
  ext.insertField("DETY", "1E", "mm", TFLOAT);
  ext.insertField("BARX", "1E", "mm", TFLOAT);
  ext.insertField("BARY", "1E", "mm", TFLOAT);
  ext.insertField("TRK_M2T", "1E", "mm^2", TFLOAT);
  ext.insertField("TRK_M2L", "1E", "mm^2", TFLOAT);
  ext.insertField("TRK_M3L", "1E", "mm^3", TFLOAT);
  ext.insertField("TRK_SKEW", "1E", "", TFLOAT);
}

/*!
 */
void ixpeFitsDataFormat::addLvl1aEventsVariableSizeFields(
    ixpeFitsBinaryTableExtension& ext)
{
  //ext.insertField("PIX_TRK", "QI", "", TSHORT);
  //ext.insertField("PIX_PIS", "QI", "", TSHORT);
  ext.insertField("PIX_X", "400E", " ", TFLOAT);
  ext.insertField("PIX_Y", "400E", " ", TFLOAT);
  ext.insertField("PIX_PHA", "400I", " ", TSHORT);
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvl1a "EVENTS" binary
  table extension.
*/
ixpeFitsBinaryTableExtension
ixpeFitsDataFormat::ixpeLvl1aEventsExtension(bool includeVariableLengthFields)
{
  ixpeFitsBinaryTableExtension ext (
      ixpeFitsDataFormat::ixpeLvl1aEventsExtensionHeader());
  ixpeFitsDataFormat::addLvl1aEventsFixedSizeFields(ext);
  if (includeVariableLengthFields) {
    ixpeFitsDataFormat::addLvl1aEventsVariableSizeFields(ext);
  }
  return ext;
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvl1a "GTI" binary table
  extension.
*/
ixpeFitsBinaryTableExtension ixpeFitsDataFormat::ixpeLvl1aGtiExtension()
{
  ixpeFitsBinaryTableExtension ext(
      ixpeFitsDataFormat::ixpeLvl1aGtiExtensionHeader());
  // This extension is identical to the corresponding LVL1 one
  ixpeFitsDataFormat::addLvl1GtiFields(ext);
  return ext;
}

/*!
  Declaration of the fields (a.k.a. columns) in the Lvla "MONTE_CARLO"
  binary table extension.
*/
ixpeFitsBinaryTableExtension ixpeFitsDataFormat::ixpeLvl1aMcExtension()
{
  ixpeFitsBinaryTableExtension ext (
      ixpeFitsDataFormat::ixpeLvl1aMcExtensionHeader());
  // Note that, differently from the Lvl1 MonteCarlo extension, we do not add
  // the fields with variable length
  ixpeFitsDataFormat::addLvl1McFixedSizeFields(ext);
  return ext;
}
