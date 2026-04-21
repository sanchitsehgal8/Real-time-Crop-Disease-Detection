export function Footer() {
  return (
    <footer className="bg-[#1a3a2a] px-6 py-6 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="text-center md:text-left">
          <p className="text-white font-semibold tracking-wide text-sm">
            AGROVISION
          </p>
          <p className="text-gray-400 text-sm">
            © 2024 AgroVision. The Digital Greenhouse Architecture.
          </p>
        </div>

        <nav className="flex flex-wrap items-center justify-center gap-4 md:gap-6">
          <a 
            href="#" 
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            Sustainability Report
          </a>
          <a 
            href="#" 
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            API Access
          </a>
          <a 
            href="#" 
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            Network Status
          </a>
          <a 
            href="#" 
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            Legal
          </a>
        </nav>
      </div>
    </footer>
  )
}
