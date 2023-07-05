import Handsontable from "handsontable";
import Axios from 'axios'
import "handsontable/dist/handsontable.min.css";
import "pikaday/css/pikaday.css";

// import { data } from "./constants";
import { progressBarRenderer, starRenderer } from "./customRenderers";

import {
  alignHeaders,
  addClassesToRows,
  changeCheckboxCell
} from "./hooksCallbacks";

const example = document.getElementById("handsontable");
const saveButton = document.getElementById("saveBtn")
console.log('this file is being run')
Axios.get('https://david-project.onrender.com/get-csv-data').then(r => {
        // the data gotten will update as so 
        //data == r.data
       console.log(r.data)
       const hot = new Handsontable(example, {
            data: r.data.data,
            height: 1000,
            colWidths: [140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140],
            colHeaders: [
              "Depth (ft)",
              "Pressure(Psi)",
              "Test flow rate(MCF/D)",
              "Compressibility",
              "Density of gas",
              "Tubing ID(in)",
              "Casing ID(in)",
              "Tubing OD(in)",
              "Flow Area",
              "Density of Liquid",
              "Temperature",
              "Surface Tension",
          ],
            columns: [
              { data: 0, type: "numeric" },
              { data: 1, type: "numeric" },
              { data: 2, type: "numeric" },
              { data: 3, type: "numeric" },
              { data: 4, type: "numeric" },
              { data: 5, type: "numeric" },
              { data: 6, type: "numeric" },
              { data: 7, type: "numeric" },
              { data: 8, type: "numeric" },
              { data: 9, type: "numeric" },
              { data: 10, type: "numeric" },
              { data: 11, type: "numeric" },
            ],
            dropdownMenu: true,
            hiddenColumns: {
              indicators: true
            },
            contextMenu: true,
            multiColumnSorting: true,
            filters: true,
            rowHeaders: true,
            manualRowMove: true,
            afterGetColHeader: alignHeaders,
            afterOnCellMouseDown: changeCheckboxCell,
            beforeRenderer: addClassesToRows,
            licenseKey: "non-commercial-and-evaluation"
          });

        saveButton.addEventListener('click', () => {
            // save all cell's data
            console.log(hot.getData())
            Axios.post('https://david-project.onrender.com/update-csv', { data: hot.getData() } ).then(() => {
                console.log('The POST request is only used here for the demo purposes');
            });
        });
    });
  
document.onload = function (){
    console.log('this part of the code was executed')
  }


